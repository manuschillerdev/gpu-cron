import { DAY_NAMES, MONTH_NAMES } from './features.js';
import type { Diagnostic, Family, Schedule } from './types.js';
import type { TokenPrediction } from './model/parameters.js';

const ALL_MONTHS = Array.from({ length: 12 }, (_, i) => i + 1);
const ALL_DAYS = Array.from({ length: 7 }, (_, i) => i);
const range = (count: number, start = 0) => Array.from({ length: count }, (_, i) => start + i);
const unique = (values: number[]) => [...new Set(values)].sort((a, b) => a - b);
class CompileError extends Error {
  constructor(public code: 'unsupported-syntax' | 'invalid-value', message: string) { super(message); }
}
const unsupported = (message: string): never => { throw new CompileError('unsupported-syntax', message); };
const invalid = (message: string): never => { throw new CompileError('invalid-value', message); };
const cardinal = 'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty'.split(' ');
const ordinal = 'zeroth first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth twentieth'.split(' ');
const values: Record<string, number> = Object.fromEntries([...cardinal.map((w,i)=>[w,i]), ...ordinal.map((w,i)=>[w,i]), ['thirty',30], ['thirtieth',30], ['forty',40], ['fifty',50], ['quarter',15], ['half',30], ['noon',12], ['midnight',0], ['last',-1], ['end',-1]]);
const word = (t:TokenPrediction)=>t.text.toLowerCase();

/** Decode only values identified by the network; never recover missing roles from text. */
function number(ts:TokenPrediction[]):number {
  if (!ts.length) unsupported('The model did not identify a required numeric value.');
  let total=0;
  for (let i=0;i<ts.length;i++) {
    const w=word(ts[i]!);
    const n=/^\d+$/.test(w)?Number(w):values[w];
    if(n===undefined) unsupported(`Cannot interpret the predicted numeric value "${w}".`);
    if(i && !(total>=20 && total%10===0 && n!>=1 && n!<=9 && !/^\d+$/.test(w))) unsupported('The model predicted more than one numeric value for a field.');
    total+=n!;
  }
  return total;
}
function named(t:TokenPrediction,months=false):number[] {
  const w=word(t);
  if(!months && /^weekdays?$/.test(w))return [1,2,3,4,5];
  if(!months && /^weekends?$/.test(w))return [0,6];
  const names=months?MONTH_NAMES:DAY_NAMES;
  const i=names.findIndex(n=>w===n || w===n.slice(0,3) || w===n+'s');
  if(i<0)unsupported(`Cannot interpret the predicted ${months?'month':'weekday'} "${w}".`);
  return [i+(months?1:0)];
}
function namedSet(tokens:TokenPrediction[],role:TokenPrediction['role'],months=false,consumed=new Set<TokenPrediction>()):number[] {
  const result:number[]=[]; const selected=tokens.map((t,i)=>({t,i})).filter(({t})=>t.role===role);
  for(let j=0;j<selected.length;j++) {
    const {t,i}=selected[j]!, first=named(t,months);result.push(...first);
    const next=selected[j+1];
    if(next && tokens.slice(i+1,next.i).some(t=>t.role==='range')) {
      const last=named(next.t,months);
      tokens.slice(i+1,next.i).filter(t=>t.role==='range').forEach(t=>consumed.add(t));
      if(first.length!==1 || last.length!==1)unsupported('Ranges require individual day or month endpoints.');
      for(let n=first[0]!, count=0;n!==last[0]! && count<12;count++) {
        n=months?n%12+1:(n+1)%7;result.push(n);
      }
      j++;
    }
  }
  return unique(result);
}

export function compile(text: string, family: Family, timeZone: string, anchorWeek: string, tokens: TokenPrediction[]): { schedule: Schedule | null; diagnostics: Diagnostic[] } {
  const diagnostics:Diagnostic[]=[];
  try {
    if(family==='invalid' || !tokens.length || tokens.some(t=>t.role==='unknown'))unsupported('The model identified unsupported or unresolved schedule language.');
    const consumedRanges=new Set<TokenPrediction>();
    const get=(role:TokenPrediction['role'])=>tokens.filter(t=>t.role===role);
    if(family!=='monthly' && get('monthday').length)unsupported('Predicted month days conflict with the recurrence family.');
    if(family!=='minutes' && family!=='hours' && get('quantity').length)unsupported('An interval quantity conflicts with the clock-based recurrence.');
    const schedule:Schedule={family:family as Exclude<Family,'invalid'>,minutes:[0],hours:[0],daysOfMonth:null,weekdays:null,months:[...ALL_MONTHS],weekInterval:family==='biweekly'?2:1,anchorWeek,timeZone};
    if(family==='minutes' || family==='hours') {
      if(get('hour').length || get('minute').length || get('meridiem').length || get('clock-offset').length)unsupported('An interval with an additional clock restriction is not supported.');
      const quantity=get('quantity');const interval=quantity.length?number(quantity):1;
      const field=family==='minutes'?60:24;
      if(interval<1 || interval>field || field%interval!==0)invalid(`The interval must divide ${field} evenly to preserve spacing across cron field boundaries.`);
      schedule.minutes=family==='minutes'?range(60).filter(n=>n%interval===0):[0];
      schedule.hours=family==='hours'?range(24).filter(n=>n%interval===0):range(24);
    }else {
      const hourTokens=get('hour'), minuteTokens=get('minute'), offsets=get('clock-offset'), directions=get('clock-direction');
      let hour=number(hourTokens), minute=minuteTokens.length?number(minuteTokens):0;
      const meridiem=get('meridiem').map(word).filter(w=>w!=='m');
      if(meridiem.length>1)unsupported('More than one meridiem was predicted.');
      if(hour<0 || hour>23 || minute<0 || minute>59)invalid('Clock time is out of range.');
      if(meridiem.length) {
        if(hour<1 || hour>12)invalid('A clock with AM/PM must have an hour between 1 and 12.');
        const pm=['pm','p','afternoon','evening','night'].includes(meridiem[0]!);
        hour=hour%12+(pm?12:0);
      }
      if(offsets.length) {
        if(minuteTokens.length || directions.length!==1)unsupported('Clock offsets need one direction and no separate minute field.');
        const offset=number(offsets);if(offset<1 || offset>59)invalid('Clock offset is out of range.');
        const before=['to','before'].includes(word(directions[0]!));
        const clock=(hour*60+(before?-offset:offset)+1440)%1440;hour=Math.floor(clock/60);minute=clock%60;
      }else if(directions.length)unsupported('Clock direction has no offset.');
      schedule.hours=[hour];schedule.minutes=[minute];
      if(!meridiem.length && !minuteTokens.length && !offsets.length && !hourTokens.some(t=>['noon','midnight'].includes(word(t))))diagnostics.push({code:'assumption',severity:'info',message:`Interpreted the predicted hour as ${String(hour).padStart(2,'0')}:00 using a 24-hour clock.`});
      if(family==='weekly' || family==='biweekly') {
        const days=namedSet(tokens,'weekday',false,consumedRanges);if(!days.length)unsupported('The model did not identify scheduled weekdays.');schedule.weekdays=days;
      }else if(family==='daily' && get('weekday').length)unsupported('Daily family conflicts with predicted weekday restrictions.');
      if(family==='monthly') {
        if(get('weekday').length)unsupported('A monthly ordinal combined with a weekday requires an unsupported nth-weekday rule.');
        const days=get('monthday');if(!days.length)unsupported('The model did not identify a day of the month.');
        // Separate list elements by original token adjacency; compound spoken numbers stay together.
        const groups:TokenPrediction[][]=[];
        for(const t of days){const last=groups.at(-1)?.at(-1);if(last && /^\s+$/.test(text.slice(last.end,t.start)) && !['last','end'].includes(word(t)))groups.at(-1)!.push(t);else groups.push([t]);}
        schedule.daysOfMonth=unique(groups.map(number));
        if(schedule.daysOfMonth.some(d=>d!==-1&&(d<1||d>31)))invalid('Day of month must be between 1 and 31, or last.');
      }
      if(family==='biweekly')diagnostics.push({code:'assumption',severity:'info',message:`Alternate weeks are anchored to the local week starting ${anchorWeek} (Monday). Preserve anchorWeek when reusing this schedule.`});
    }
    const included=namedSet(tokens,'month',true,consumedRanges);if(included.length)schedule.months=included;
    const excludedMonths=namedSet(tokens,'excluded-month',true,consumedRanges), excludedDays=namedSet(tokens,'excluded-weekday',false,consumedRanges);
    if((excludedMonths.length || excludedDays.length) && !get('exclusion').length)unsupported('An excluded value has no predicted exclusion marker.');
    if(get('exclusion').length && !excludedMonths.length && !excludedDays.length)unsupported('The model identified an exclusion without a supported target.');
    schedule.months=schedule.months.filter(m=>!excludedMonths.includes(m));
    if(!schedule.months.length)invalid('The exclusions remove every month.');
    if(excludedDays.length){schedule.weekdays=(schedule.weekdays??ALL_DAYS).filter(d=>!excludedDays.includes(d));if(!schedule.weekdays.length)invalid('The exclusions remove every scheduled weekday.');}
    if(get('range').some(t=>!consumedRanges.has(t)))unsupported('A predicted range has no matching day or month endpoints.');
    return {schedule,diagnostics};
  }catch(error){if(!(error instanceof CompileError))throw error;return {schedule:null,diagnostics:[...diagnostics,{code:error.code,severity:'error',message:error.message}]};}
}

function field(values: number[], count: number, start = 0): string {
  if (values.length === count) return '*';
  // Only emit steps for complete, zero-based field partitions.
  if (start === 0 && values.length > 1 && values[0] === 0) {
    const step = values[1]!;
    if (count % step === 0 && values.length === count / step && values.every((value, i) => value === i * step)) return `*/${step}`;
  }
  if (values.length > 2 && values.every((value, i) => value === values[0]! + i)) return `${values[0]}-${values.at(-1)}`;
  return values.join(',');
}

export function cron(schedule: Schedule): { expression: string | null; reason?: string } {
  if (schedule.weekInterval === 2) return { expression: null, reason: 'Five-field cron cannot preserve an every-other-week interval. Use the structured schedule and its anchorWeek.' };
  if (schedule.daysOfMonth?.includes(-1)) return { expression: null, reason: 'The last day of a month requires calendar logic; standard five-field cron has no L operator.' };
  if (schedule.daysOfMonth && schedule.weekdays) return { expression: null, reason: 'Cron combines restricted day-of-month and weekday fields with OR. This schedule requires both restrictions (AND).' };
  return { expression: [field(schedule.minutes, 60), field(schedule.hours, 24), schedule.daysOfMonth ? field(schedule.daysOfMonth, 31, 1) : '*', field(schedule.months, 12, 1), schedule.weekdays ? field(schedule.weekdays, 7) : '*'].join(' ') };
}

export function describe(schedule: Schedule): string {
  const time = `${String(schedule.hours[0]).padStart(2, '0')}:${String(schedule.minutes[0]).padStart(2, '0')}`;
  let text: string;
  if (schedule.family === 'minutes') text = `Every ${60 / schedule.minutes.length} minute${schedule.minutes.length === 60 ? '' : 's'}`;
  else if (schedule.family === 'hours') text = `Every ${24 / schedule.hours.length} hour${schedule.hours.length === 24 ? '' : 's'}, on the hour`;
  else if (schedule.daysOfMonth) text = `${schedule.daysOfMonth[0] === -1 ? 'The last day' : `Day ${schedule.daysOfMonth.join(' and ')}`} of each month at ${time}`;
  else if (schedule.weekdays) text = `${schedule.weekInterval === 2 ? 'Every other week on' : 'Every'} ${schedule.weekdays.map(day => DAY_NAMES[day]).join(', ')} at ${time}`;
  else text = `Every day at ${time}`;
  if ((schedule.family === 'minutes' || schedule.family === 'hours' || schedule.daysOfMonth) && schedule.weekdays) text += `, on ${schedule.weekdays.map(day => DAY_NAMES[day]).join(', ')}`;
  if (schedule.months.length !== 12) text += `, except ${ALL_MONTHS.filter(month => !schedule.months.includes(month)).map(month => MONTH_NAMES[month - 1]).join(', ')}`;
  return `${text} · ${schedule.timeZone}`;
}
