"""Reproduce analysis, tests, six reference scenarios, and a linked HTML report."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

from prepare import ROOT, Preparation, save_json
from analyze import analyze
from simulate import experiment
from report import render


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'results/rebuild')
    p.add_argument('--from-originals',action='store_true',help='Re-extract local private copies; not needed for prepared-data reproduction')
    p.add_argument('--config',type=Path,default=ROOT/'config.json')
    p.add_argument('--smoke',action='store_true',help='3 replications/scenario; no bootstrap p-values')
    a=p.parse_args();out=a.output.resolve()
    if out.exists() and any(out.iterdir()):p.error('Output directory must be new or empty; choose a new --output.')
    out.mkdir(parents=True,exist_ok=True)
    if a.from_originals:Preparation(ROOT/'private_originals').run(ROOT/'data')
    config=json.loads(a.config.read_text())
    if a.smoke:config['replications']=3
    test=subprocess.run([sys.executable,'-B','-m','pytest','-q','-p','no:cacheprovider'],cwd=ROOT,text=True,capture_output=True)
    (out/'tests.txt').write_text(test.stdout+test.stderr)
    if test.returncode:raise RuntimeError(f'Tests failed; see {out / "tests.txt"}')
    analyze(ROOT/'data',out/'input',bootstrap=0 if a.smoke else 199,seed=config['seed'])
    experiment(ROOT/'data',out/'simulation',config)
    runtime_files=['prepare.py','analyze.py','simulate.py','report.py','run.py']
    code={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in runtime_files}
    code.update({str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((ROOT/'tests').glob('*.py'))})
    save_json(out/'run.json',{'timestamp_utc':datetime.now(timezone.utc).isoformat(),'python':platform.python_version(),
        'platform':platform.platform(),'dependencies':{n:importlib.metadata.version(n) for n in ['numpy','scipy','openpyxl','matplotlib','pytest']},
        'mode':'smoke' if a.smoke else 'full_reference_experiment','config':config,'code_sha256':code,
        'data_manifest':json.loads((ROOT/'data/manifest.json').read_text()),
        'tests':{'returncode':test.returncode,'summary':test.stdout.strip().splitlines()[-1]},
        'anylogic_compiled':False,'anylogic_executed':False})
    print(render(out))


if __name__=='__main__':main()
