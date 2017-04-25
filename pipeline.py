#!/usr/bin/env python3

# Beeldbank vertaler pipeline
#  Maarten van Gompel
#  GPLv3

import sys
import os
import json
import argparse
import ucto

def annotations(data, quiet=False):
    for annotation in data['annotations']:
        annotation['caption'] = annotation['caption'].replace('\r','')
        if '\n' in annotation['caption'].strip():
            if not quiet: print("WARNING: Skipping caption containing newline: " + repr(annotation), file=sys.stderr)
        elif not annotation['caption'].strip():
            if not quiet: print("WARNING: Missing/empty caption: " + repr(annotation), file=sys.stderr)
        else:
            yield annotation


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #parser.add_argument('--storeconst',dest='settype',help="", action='store_const',const='somevalue')
    parser.add_argument('--mosesini', type=str,help="Path to moses.ini", action='store',default="",required=False)
    parser.add_argument('--threads',type=int,help="Threads", action='store',default=1,required=False)
    parser.add_argument('inputfiles', nargs='+', help='Input files')
    args = parser.parse_args()

    for jsonfilename in args.inputfiles:
        basename = os.path.basename(jsonfilename).replace('.json','')
        print("Loading " + basename + " ...",file=sys.stderr)
        data = json.load(open(jsonfilename,'r',encoding='utf-8'))

        if not os.path.exists(basename+'.txt') or not os.path.exists(basename+'.idx'):
            print("Extracting data from " + basename + " ...",file=sys.stderr)
            with open(basename+'.txt','w',encoding='utf-8') as sentenceoutput, open(basename+'.idx','w') as indexfile:
                for annotation in annotations(data):
                    print(annotation['caption'].strip(), file=sentenceoutput)
                    print(annotation['id'], file=indexfile)

        if not os.path.exists(basename+'.tok'):
            print("Tokenizing data for " + basename + " ...",file=sys.stderr)
            tokenizer = ucto.Tokenizer('tokconfig-eng', sentenceperlineinput=True, sentenceperlineoutput=True,sentencedetection=False, paragraphdetection=False)
            tokenizer.tokenize(basename + '.txt', basename + '.tok')

        if not os.path.exists(basename+'.out'):
            print("Translating " + basename + " ...",file=sys.stderr)
            os.system("moses --threads " + str(args.threads) + " -f " + args.mosesini + " < " + basename +'.tok > ' + basename + '.out')

        if not os.path.exists(basename + '.translated.json'):
            print("Consolidating final output for " + basename + " ...",file=sys.stderr)
            with open(basename+'.out','r',encoding='utf-8') as outputfile, open(basename+'.idx','r') as indexfile:
                translations = {}
                for outline, idline in zip(outputfile.readlines(), indexfile.readlines()):
                    id = idline.strip()
                    translations[id] = outline.strip()

                for annotation in annotations(data, quiet=True):
                    id = str(annotation['id'])
                    if id in translations:
                        annotation['caption_nl'] = translations[id]
                    else:
                        print("WARNING: No translation for " + id + " !!",file=sys.stderr)

            with open(basename + '.translated.json','w',encoding='utf-8') as jsonout:
                json.dump(data, jsonout, indent=1)
