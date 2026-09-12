"""Offline teacher interchange: export prompts, import annotated proposals.

No API keys, model downloads, or teacher dependencies in the deployed package.
The current assistant can supply the JSONL, or a separately run local teacher can.
"""
import argparse
import json
from pathlib import Path
from data import DATA, datasets, record

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prompts','import'])
    parser.add_argument('--file',type=Path)
    parser.add_argument('--model',default='assistant')
    parser.add_argument('--limit',type=int,default=20)
    args=parser.parse_args()
    parents={r['id']:r for r in datasets()['train'] if r['source']=='generator'}
    if args.command=='prompts':
        chosen=[];groups=set()
        for parent in parents.values():
            if parent['target']['status']!='supported' or parent['groupId'] in groups:continue
            groups.add(parent['groupId']);chosen.append(parent)
            if len(chosen)>=args.limit:break
        for parent in chosen:
            print(json.dumps({'parentId':parent['id'],'text':parent['text'],'meaning':parent['target'],
                'instruction':'Propose one natural paraphrase with exactly this meaning. Preserve interval, clock, weekdays, month days and exclusions. Do not change every-other-week to weekly. Return JSON {parentId, category, parts:[{text,role}]}, with roles prefix/recurrence/time/exclusion/unknown and spaces included. Do not infer labels from the tiny model.'}))
        return
    if args.file is None:parser.error('import requires --file')
    proposals=[json.loads(line) for line in args.file.read_text().splitlines() if line.strip()]
    path=DATA/'paraphrases.jsonl'
    admitted=[json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    by_id={r['id']:r for r in admitted}
    parent_path=DATA/'paraphrase-parents.jsonl'
    frozen={r['id']:r for r in map(json.loads,parent_path.read_text().splitlines())} if parent_path.exists() else {}
    for proposal in proposals:
        parent=parents[proposal['parentId']]
        frozen[parent['id']]=parent
        if 'target' in proposal and proposal['target']!=parent['target']:raise ValueError('Teacher must not replace the parent meaning')
        parts=[(p['text'],p['role']) for p in proposal['parts']]
        if any(role not in ['prefix','recurrence','time','exclusion','unknown'] for _,role in parts):raise ValueError('Unknown token role')
        item=record(parts,parent['family'],parent['target'],'teacher-paraphrase',proposal['category'],'assistant-paraphrase')
        item.update(parentId=parent['id'],teacher=args.model,split='train')
        by_id[item['id']]=item
    path.write_text(''.join(json.dumps(item,ensure_ascii=False)+'\n' for item in by_id.values()))
    parent_path.write_text(''.join(json.dumps(item,ensure_ascii=False)+'\n' for item in frozen.values()))
    print(json.dumps({'imported':len(proposals),'total':len(by_id),'note':'Parent meaning/group inherited. Schema checks do not prove semantic fidelity; provenance remains visible. No review gate.'}))

if __name__=='__main__':main()
