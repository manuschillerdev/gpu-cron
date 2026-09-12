"""Meaning-first schedule data. Rendering never calls the runtime compiler."""
import hashlib
import json
import random
import re
from pathlib import Path
import numpy as np

VERSION = 12
SEED = 7319
VOCAB = 1312
WIDTH = 24
MAX_TOKENS = 512
FAMILIES = ['minutes', 'hours', 'daily', 'weekly', 'biweekly', 'monthly', 'invalid']
ROLES = ['prefix', 'recurrence', 'quantity', 'hour', 'minute', 'meridiem', 'clock-offset', 'clock-direction', 'weekday', 'range', 'monthday', 'month', 'excluded-weekday', 'excluded-month', 'exclusion', 'unknown']
DAYS = 'sunday monday tuesday wednesday thursday friday saturday'.split()
MONTHS = 'january february march april may june july august september october november december'.split()
NUMBERS = 'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty'.split()
DATA = Path(__file__).resolve().parent / 'data'
VOCABULARY = DATA.parents[1]/'src/model/vocabulary.json'

def tokens(text):
    return list(re.finditer(r'[A-Za-z]+|[0-9]+|[^\s]', text))

def word_hash(text):
    value=2166136261
    for c in text:
        value=((value ^ ord(c))*16777619)&0xffffffff
    return value

def token_id(text):
    word=text.lower()
    kind=1 if re.fullmatch('[0-9]+',word) else 0 if re.fullmatch('[a-z]+',word) else 2
    shape=kind | (min(len(word)//3,3)<<2) | (int(text!=word)<<4)
    return 1+(word_hash(word)&1023)+((word_hash(re.sub('[aeiou]','',word))&255)<<10)+(shape<<18)

def write_vocabulary(training):
    # Kept as a provenance artifact; the runtime uses feature hashes, no word lookup.
    words=sorted({t[0].lower() for item in training for t in tokens(item['text'])})
    VOCABULARY.write_text(json.dumps(words,ensure_ascii=False,separators=(',',':'))+'\n')
    return len(words)

def features(text, length=MAX_TOKENS):
    ts=tokens(text)
    if len(ts)>length: raise ValueError('Sequence exceeds token capacity')
    return np.array([token_id(t[0]) for t in ts]+[0]*(length-len(ts)),dtype=np.float32)

ORDINALS='zeroth first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth twentieth'.split()
NUMBER_WORDS=set(NUMBERS+['thirty','forty','fifty']+ORDINALS+['thirtieth'])
def number_word(n):
    if n<=20:return NUMBERS[n]
    return {20:'twenty',30:'thirty',40:'forty',50:'fifty'}[n//10*10]+(' '+NUMBERS[n%10] if n%10 else '')

def semantic_labels(item):
    """Refine existing authored clause annotations; never use model/compiler output.

    Generator records pass through the same deterministic annotation convention.
    Original clause labels remain recorded to make the migration auditable.
    """
    labels=item['tokens'];text=item['text'];words=[text[t['start']:t['end']].lower() for t in labels]
    coarse=[t.get('clause',t['role']) for t in labels]
    time_indices=[i for i,c in enumerate(coarse) if c=='time']
    direction=next((i for i in time_indices if words[i] in ('past','to','before','after')),None)
    numbers=[i for i in time_indices if words[i].isdigit() or words[i] in NUMBER_WORDS or words[i] in ('noon','midnight')]
    colon=next((i for i in time_indices if words[i] in (':','.') and i and i+1<len(words) and words[i-1].isdigit() and words[i+1].isdigit()),None)
    for i,t in enumerate(labels):
        w=words[i];c=coarse[i];role=c
        if c=='unknown':
            # A restriction can be unsupported while its ordinary words keep their roles.
            if w in ('every','each','daily','hourly','day','days','minute','minutes','hour','hours','week','weeks','monthly'):role='recurrence'
            elif any(w in (d,d[:3],d+'s') for d in DAYS) or w in ('weekday','weekdays','weekend','weekends'):role='weekday'
            elif w in ('please','run','it','this','the','a','an','at','on','in','of',',','.','!','?'):role='prefix'
        elif c=='time':
            role='prefix'
            if w in ('am','pm','morning','afternoon','evening','night') or (w in ('a','p','m') and '.' in words[max(0,i-1):i+3]):role='meridiem'
            elif w in ('quarter','half'):role='clock-offset'
            elif i==direction:role='clock-direction'
            elif i in numbers:
                role='clock-offset' if direction is not None and i<direction else 'minute' if colon is not None and i>colon else 'hour'
        elif c in ('recurrence','exclusion'):
            excluded=c=='exclusion'
            day=any(w in (d,d[:3],d+'s') for d in DAYS) or w in ('weekday','weekdays','weekend','weekends')
            month=any(w in (m,m[:3]) for m in MONTHS)
            if day:role='excluded-weekday' if excluded else 'weekday'
            elif month:role='excluded-month' if excluded else 'month'
            elif w in ('through','to','-','–','—'):role='range' if not (c=='recurrence' and item['family'] in ('minutes','hours')) else 'recurrence'
            elif not excluded and item['family']=='monthly' and (w.isdigit() or w in NUMBER_WORDS or w in ('last','end')):role='monthday'
            elif not excluded and item['family'] in ('minutes','hours') and (w.isdigit() or w in NUMBER_WORDS or w in ('half','quarter','minus')):role='quantity'
            elif w in (',','/','&','+','(',')','.', '!', '?') or w in ('and','or','the','a','an','in','during','on','of','at','s','st','nd','rd','th'):role='prefix'
        t['clause']=c;t['role']=role
    item['annotationVersion']=2
    return item

def schedule(family, hour=9, minute=0, weekdays=None, monthdays=None, months=None,
             interval=1, excluded_weekdays=()):
    days = sorted(set(weekdays)) if weekdays is not None else None
    if excluded_weekdays:
        days = sorted(set(range(7) if days is None else days)-set(excluded_weekdays))
    return {'family':family,'minutes':list(range(0,60,interval)) if family=='minutes' else [minute if family!='hours' else 0],
            'hours':list(range(24)) if family=='minutes' else list(range(0,24,interval)) if family=='hours' else [hour],
            'daysOfMonth':sorted(set(monthdays)) if monthdays is not None else None,'weekdays':days,
            'months':sorted(set(months)) if months is not None else list(range(1,13)),
            'weekInterval':2 if family=='biweekly' else 1}

def group_id(target):
    # Paraphrases with identical supported meaning share a group, regardless of source.
    semantic = {k:v for k,v in target['schedule'].items() if k!='family'} if target['status']=='supported' else target
    return hashlib.sha256(json.dumps(semantic,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:20]

def record(parts, family, target, pattern, category, source='generator', identity=None):
    text=''.join(part for part,_ in parts)
    labels=[];cursor=0
    for part,role in parts:
        for token in tokens(part):
            labels.append({'start':cursor+token.start(),'end':cursor+token.end(),'role':role})
        cursor+=len(part)
    if [(t.start(),t.end()) for t in tokens(text)] != [(t['start'],t['end']) for t in labels]:
        raise ValueError('A labelled piece split a token')
    group=group_id(target)
    return semantic_labels({'id':identity or hashlib.sha256((group+'|'+text).encode()).hexdigest()[:20],
            'groupId':group,'text':text,'family':family,'tokens':labels,'target':target,
            'pattern':pattern,'category':category,'source':source})

def row(item):
    return item['text'], FAMILIES.index(item['family']), [ROLES.index(t['role']) for t in item['tokens']]

def authored():
    path=DATA/'authored.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    for item in rows:
        item['groupId']=group_id(item['target'])
        semantic_labels(item)
    return rows

def meaning(rng, index):
    family=FAMILIES[index%7]
    if family=='invalid':
        kind=rng.choice(['relative','event','vague','annual','nth-weekday','bounded','multi-clock','business','timezone','non-request','uncertain','invalid-value'])
        return {'family':family,'kind':kind,'target':{'status':'unsupported','schedule':None,'reason':kind}}
    hour=rng.choice([0,12]) if rng.random()<.2 else rng.randrange(24)
    minute=rng.choice([0,0,15,30,45]) if rng.random()<.7 else rng.randrange(60)
    interval=rng.choice([1,2,3,4,5,6,10,12,15,20,30]) if family=='minutes' else rng.choice([1,2,3,4,6,8,12])
    weekdays=sorted(rng.sample(range(7),rng.choice([1,1,2,3,5]))) if family in ('weekly','biweekly') else None
    if family=='weekly' and rng.random()<.35:weekdays=rng.choice([[1,2,3,4,5],[0,6],[0,5,6]])
    monthdays=rng.choice([[-1],[1],[1,15],[rng.randint(1,28)]]) if family=='monthly' else None
    excluded_months=sorted(rng.sample(range(1,13),rng.choice([1,1,2,3,4]))) if rng.random()<.35 else []
    excluded_days=rng.choice([[0,6],[rng.randrange(7)]]) if not excluded_months and rng.random()<.15 else []
    if weekdays is not None and not set(weekdays)-set(excluded_days):excluded_days=[]
    target={'status':'supported','schedule':schedule(family,hour,minute,weekdays,monthdays,[m for m in range(1,13) if m not in excluded_months],interval,excluded_days)}
    return dict(family=family,hour=hour,minute=minute,interval=interval,weekdays=weekdays,monthdays=monthdays,excluded_months=excluded_months,excluded_days=excluded_days,target=target)

def ordinal(n):
    return str(n)+('th' if 10<n%100<14 else {1:'st',2:'nd',3:'rd'}.get(n%10,'th'))

def spoken_clock(hour, minute, rng):
    def hour_text(h):
        if h%24==0:return 'midnight'
        if h%24==12:return 'noon'
        return number_word(h%12 or 12)+(' in the morning' if h%24<12 else rng.choice([' in the afternoon',' in the evening',' at night']))
    if minute==0:return hour_text(hour)
    before=rng.random()<.5
    offset=60-minute if before else minute
    base=(hour+1)%24 if before else hour
    amount=rng.choice(['a quarter','quarter',number_word(offset)]) if offset==15 else rng.choice(['half','thirty']) if offset==30 else number_word(offset)
    return amount+(' ' if 'quarter' in amount or amount=='half' else rng.choice([' minutes ',' ']))+rng.choice(['to','before'] if before else ['past','after'])+' '+hour_text(base)

def render(m, rng, pattern):
    """Composable surface forms, all drawn from a fixed structured meaning."""
    f=m['family']
    if f=='invalid':
        phrases={'relative':['tomorrow at noon','in five minutes','next friday'],
                 'event':['when the server starts','after deployment','when the build finishes'],
                 'vague':['sometime soon','after work','when convenient'],
                 'annual':['once a year','every leap year','annually'],
                 'nth-weekday':[f'every {rng.choice(ORDINALS[1:5])} {rng.choice(DAYS)} of the month at noon', 'on the second friday each month','every last tuesday of the month'],
                 'bounded':[f'every {rng.choice([5,10,15])} minutes between {rng.randrange(1,12)}am and 5pm',f'daily at {rng.randrange(24)}:00 until friday','every hour for the next two days','every friday starting next week'],
                 'multi-clock':[f'every day at {rng.randrange(1,12)}am and {rng.randrange(1,12)}pm','on monday at noon and friday at midnight'],
                 'business':[f'every {rng.choice(DAYS)} at noon except holidays','daily at noon unless the office is closed','every first business day of the month'],
                 'timezone':[f'daily at {rng.randrange(24)}:00 {rng.choice(["UTC","Europe/Berlin","New York time","Pacific time"])}'],
                 'uncertain':['every few hours should be fine','weekdays around nine','daily at eight-ish','every trading day at noon','every minus two hours','weekly at half five'],
                 'invalid-value':['every zero minutes','every 0 hours','every 25 hours','daily at 25:00'],
                 'non-request':['what does this schedule mean','please cancel the friday job','do not run anything','every day is different','I might need an hourly job later']}
        if rng.random()<.5:phrases[m['kind']]=['please '+t for t in phrases[m['kind']]]
        kind=m['kind']
        if kind=='invalid-value':
            family=rng.choice(['minutes','hours','daily'])
            parts=[(f'every {rng.choice([0,7,13,17,25,61])} minutes','recurrence')] if family=='minutes' else [(f'every {rng.choice([0,5,7,25,48])} hours','recurrence')] if family=='hours' else [('daily ','recurrence'),(f'at {rng.choice([25,26,30])}:00','time')]
            return record(parts,family,m['target'],pattern,kind)
        if kind=='nth-weekday':
            return record([(rng.choice(['every ','on the ','each month on the ']),'recurrence'),(rng.choice(ORDINALS[1:5]+['last'])+' ','recurrence'),(rng.choice(DAYS)+' ','recurrence'),(rng.choice(['of the month ','of each month ','in every month ']),'recurrence'),(f'at {rng.randrange(24):02d}:00','time')], 'monthly',m['target'],pattern,kind)
        if kind in ('business','bounded','timezone','multi-clock'):
            hour=rng.randrange(24)
            endings={'business':['except public holidays','unless the office is closed'], 'bounded':['until friday','starting next week','for the next five occurrences','for thirty days'], 'timezone':['UTC','Europe/Berlin','Pacific time'], 'multi-clock':[f'and {rng.randrange(1,12)}pm']}
            item=record([(rng.choice(['daily ','every day ','each day ']),'recurrence'),(f'at {hour:02d}:00 ','time'),(rng.choice(endings[kind]),'unknown')], 'daily',m['target'],pattern,kind)
            return item
        return record([(rng.choice(phrases[kind]),'unknown')],f,m['target'],pattern,kind)
    n=m['interval'];word=NUMBERS[n] if n<len(NUMBERS) and rng.random()<.3 else str(n)
    if f in ('minutes','hours'):
        unit=rng.choice(['minute','min'] if f=='minutes' else ['hour','hr'])
        rec='hourly' if f=='hours' and n==1 and rng.random()<.5 else f'every {word+" " if n!=1 or rng.random()<.5 else ""}{unit}{"s" if n!=1 else ""}'
    elif f=='daily':rec=rng.choice(['daily','every day','every night','each day','once a day','every single day'])
    elif f in ('weekly','biweekly'):
        days=m['weekdays']
        names=[rng.choice([DAYS[d][:3], DAYS[d], DAYS[d]+'s']) for d in days]
        day_text=rng.choice([' and ', ', ', ' / ', ' & ']).join(names)
        if days==[1,2,3,4,5]:day_text=rng.choice(['weekday','weekdays','monday through friday','monday to friday'])
        if days==[0,6]:day_text=rng.choice(['weekend','weekends','saturday and sunday'])
        if days==[0,5,6]:day_text=rng.choice(['friday through sunday','fri-sun','fri–sun','friday to sunday'])
        if len(days)>1 and days==list(range(days[0],days[-1]+1)) and rng.random()<.5:
            day_text=DAYS[days[0]]+rng.choice([' through ',' to ','-','–'])+DAYS[days[-1]]
        rec=rng.choice(['every ','each ','on ','','every week on ','weekly on '])+day_text if f=='weekly' else rng.choice(['every other ','every second ','alternate ','fortnightly on ','every two weeks on ','every 2 weeks on ','every other week on '])+day_text
    else:
        days=m['monthdays'];day_text='the last day' if days==[-1] else 'the '+rng.choice([' and ', ' + ', ', ']).join(ordinal(d) for d in days)
        if days==[1] and rng.random()<.4:day_text='the first'
        if days!=[-1] and rng.random()<.5:day_text='the '+' and '.join(ORDINALS[d] if d<=20 else ordinal(d) for d in days)
        rec=rng.choice([f'on {day_text} of every month',f'every month on {day_text}',f'{day_text} of each month',f'monthly on {day_text}',f'on {day_text} of the month',f'on days '+ ' and '.join(str(d) for d in days)+' each month' if days!=[-1] else 'at month end',f'every month on day '+number_word(days[0]) if len(days)==1 and days[0]>0 else f'monthly on {day_text}'])
        if days==[-1] and rng.random()<.3:rec=rng.choice(['at month end',"on each month\'s last day"])
    if f in ('minutes','hours'):
        rec=rng.choice([rec,f'once every {word} {unit}s',f'at {word}-{unit} intervals',f'every {word} {unit}s'])
        if n==1:rec=rng.choice([rec,f'once a {unit}','hourly' if f=='hours' else 'every minute'])
        if f=='minutes' and n==30 and rng.random()<.5:rec='every half hour'
    parts=[(rec,'recurrence')]
    if f not in ('minutes','hours'):
        h=m['hour'];minute=m['minute'];h12=h%12 or 12
        clock=rng.choice([f'{h:02d}:{minute:02d}',f'{h12}:{minute:02d}{"am" if h<12 else "pm"}'])
        if minute==0:clock=rng.choice([clock,f'{h12}{"am" if h<12 else "pm"}', 'noon' if h==12 else 'midnight' if h==0 else str(h)])
        if rng.random()<.55:
            clock=spoken_clock(h,minute,rng)
        if rng.random()<.08:clock=clock.replace(':','.').replace('am','a.m.').replace('pm','p.m.')
        if pattern=='bare-clock':clock=str(h)
        time=(rng.choice(['at ','at ','','@ '])+clock,'time')
        parts=[time,(' '+rec,'recurrence')] if pattern=='clock-first' else [(rec+' ','recurrence'),time]
    excluded=m['excluded_months'];days=m['excluded_days']
    if excluded or days:
        names=[MONTHS[v-1] for v in excluded] if excluded else ['weekends'] if days==[0,6] else [DAYS[v] for v in days]
        if excluded and len(excluded)>1 and excluded==list(range(excluded[0],excluded[-1]+1)) and rng.random()<.7:
            names=[MONTHS[excluded[0]-1]+rng.choice([' through ',' to ','-','–'])+MONTHS[excluded[-1]-1]]
        intro='but not ' if pattern=='but-not' else rng.choice(['except ','excluding ','but not ','but skip ','skipping ','outside ',"unless it's ",'unless it is ','unless that day is a ','but never on '])+('in ' if excluded and rng.random()<.4 else '')
        exclusion=(intro+' and '.join(names),'exclusion')
        if rng.random()<.15:parts=[exclusion,(', ','prefix')]+parts
        elif rng.random()<.15:parts += [(' (','prefix'),exclusion,(')','prefix')]
        else:parts.append((' '+exclusion[0],exclusion[1]))
    prefix=rng.choice(['','','please ','run ','schedule ','please run ','could you schedule it ','I need it ',"I'd like it to run ",'have this run ','execute the task ','start the job ','repeat this ','trigger this ',"I'd like an ",'arrange for this to run '])
    if rng.random()<.45:prefix=''
    if prefix and rng.random()<.25:
        prefix=rng.choice(['back up','refresh','archive','email','clear','check','restart','send','wake','create'])+' '+rng.choice(['the database','the report','the logs','the service','a reminder','the cache','the dashboard','the worker','the weekly summary','the invoice report','the health endpoint'])+' '
    if pattern=='bare-clock':prefix=''
    if prefix:parts.insert(0,(prefix,'prefix'))
    if pattern!='bare-clock' and rng.random()<.25:parts.append((rng.choice(['.', '!', ', please.', ' is when I want this to run.']),'prefix'))
    if rng.random()<.08:parts=[(t.upper(),r) for t,r in parts]
    return record(parts,f,m['target'],pattern,'exclusions' if excluded or days else f)

def datasets(epoch=0):
    independent=authored()
    reserved={r['groupId'] for r in independent}
    result={'train':[],'development':[],'patternHoldout':[],'authoredHoldout':independent}
    rng=random.Random(SEED)
    surface=random.Random(SEED+epoch*1009)
    seen=set()
    for i in range(16000):
        m=meaning(rng,i);group=group_id(m['target'])
        if group in reserved or group in seen:continue
        seen.add(group)
        bucket=int(group[:8],16)%10
        split='train' if m['family']=='invalid' or bucket<8 else 'development' if bucket==8 else 'patternHoldout'
        patterns=['clock-last','clock-first','request-prefix'] if split!='patternHoldout' else ['but-not']
        if split!='patternHoldout' and m['family'] not in ('invalid','minutes','hours') and m['minute']==0:patterns.append('bare-clock')
        # Whole held-out surface construction: exclusion introduced by "but not".
        if split=='patternHoldout' and (m['family']=='invalid' or not(m['excluded_days'] or m['excluded_months'])):continue
        for pattern in patterns*(40 if m['family']=='invalid' else 1):
            item=render(m,surface if split=='train' else random.Random(SEED+i*31+len(pattern)),pattern);item['split']=split;result[split].append(item)
    # Every latent meaning belongs to one split, including external paraphrases.
    proposals=DATA/'paraphrases.jsonl'
    frozen_parents=DATA/'paraphrase-parents.jsonl'
    if frozen_parents.exists():
        present={r['id'] for r in result['train']}
        for line in frozen_parents.read_text().splitlines():
            if not line.strip():continue
            parent=semantic_labels(json.loads(line))
            group=group_id(parent['target'])
            if group!=parent['groupId'] or group in reserved or int(group[:8],16)%10>=8:
                raise ValueError('Frozen teacher parent must belong to a training meaning')
            if parent['id'] not in present:result['train'].append(parent)
    if proposals.exists():
        parents={r['id']:r for r in result['train']}
        for line in proposals.read_text().splitlines():
            if not line.strip():continue
            item=semantic_labels(json.loads(line))
            parent=parents.get(item['parentId'])
            if parent is None or item['target']!=parent['target'] or item['groupId']!=parent['groupId']:
                raise ValueError('Paraphrase must inherit a training parent meaning and split')
            result['train'].append(item)
    owners={}
    for split,items in result.items():
        for item in items:
            old=owners.setdefault(item['groupId'],split)
            if old!=split:raise ValueError(f'Meaning leaks between {old} and {split}')
    return result

def corpus(seed, count, heldout=False):
    # Compatibility for the existing benchmark; all examples come from meaning groups.
    items=datasets()['development' if heldout else 'train']
    rng=random.Random(seed);rng.shuffle(items)
    return [row(r) for r in items[:count]]

def write_manifest(sets):
    DATA.mkdir(exist_ok=True)
    counts={}
    for split,items in sets.items():
        (DATA/(('authored' if split=='authoredHoldout' else split)+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in items))
        counts[split]={'examples':len(items),'meaningGroups':len({r['groupId'] for r in items}),
                       'sha256':hashlib.sha256(json.dumps(items,sort_keys=True).encode()).hexdigest()}
    manifest={'version':VERSION,'seed':SEED,'splits':counts,'grouping':'Canonical structured meaning before rendering; authored meanings reserved from generated splits.','heldoutPattern':'Former but-not holdout is now a meaning-disjoint development stress set; the construction is taught in training.','authoredProvenance':'Assistant-authored and annotated separately from the renderer, not real-user traffic. Texts and targets are never used by trainer. Previously inspected results informed this iteration; this is development evaluation.','negativeGroups':'Unsupported generator categories all stay in training; authored evaluation supplies separate unsupported/ambiguous requests. Generated development contains supported meanings only.','policy':'The named holdout files are now development benchmarks, evaluated outside the trainer. Do not claim untouched generalization from them. A fresh collection is needed for a fresh generalization claim.'}
    (DATA/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest

if __name__=='__main__':
    print(json.dumps(write_manifest(datasets()),indent=2))
