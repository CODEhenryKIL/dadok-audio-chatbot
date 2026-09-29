"""Launch locally: python run.py [--demo] [--port 8765]."""
import argparse
import json
import os
from pathlib import Path
import uvicorn

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo', action='store_true', help='명시적 합성 육아기록 사용 (GPT/Voyage는 실제 API)')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    if args.demo:
        path = Path(__file__).parent / 'tests' / 'fixtures' / 'babylog_snapshot.synthetic.json'
        data = json.loads(path.read_text())
        os.environ['DADOK_RECORD_SNAPSHOT'] = str(path.resolve())
        os.environ['DADOK_AUTH_USER_ID'] = data['user_id']
        os.environ['DADOK_CHILD_ID'] = data['children'][0]['id']
    uvicorn.run('dadok.server:app', host='127.0.0.1', port=args.port)

if __name__ == '__main__':
    main()
