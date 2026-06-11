import json, sys, traceback

if len(sys.argv) < 2:
    print('Usage: run_first_code_cell.py path/to/notebook.ipynb')
    sys.exit(2)

nb_path = sys.argv[1]
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Ensure the notebook's parent folder (project root) is on sys.path so
# workspace-level shims like pkg_resources.py are importable.
import os
nb_parent = os.path.abspath(os.path.join(os.path.dirname(nb_path), '..'))
if nb_parent not in sys.path:
    sys.path.insert(0, nb_parent)

for i, cell in enumerate(nb.get('cells', []), start=1):
    if cell.get('cell_type') == 'code':
        print(f'Executing first code cell (cell #{i})...')
        code = ''.join(cell.get('source', []))
        try:
            exec(compile(code, '<notebook_cell>', 'exec'), globals())
        except Exception:
            traceback.print_exc()
        break
else:
    print('No code cell found in notebook.')
