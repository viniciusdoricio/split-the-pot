"""Inject the fitted model (out/model_export.json from e_generator.py) into web/tool_template.html."""
import json, os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
m = json.load(open(os.path.join(ROOT, 'out', 'model_export.json')))
t = open(os.path.join(ROOT, 'web', 'tool_template.html'), encoding='utf-8').read()
past = '|'.join('-'.join(map(str, c)) for c in m['past'])
t = (t.replace('__BETA__', json.dumps(m['beta'])).replace('__PAST__', past)
      .replace('__META__', json.dumps({'draws': len(m['past']), 'last': m['last_contest'], 'date': m['last_date']})))
open(os.path.join(ROOT, 'web', 'split_the_pot.html'), 'w', encoding='utf-8').write(t)
print('web/split_the_pot.html rebuilt with', len(m['past']), 'draws')
