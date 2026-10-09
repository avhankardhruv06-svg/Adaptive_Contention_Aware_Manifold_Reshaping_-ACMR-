"""
select_workflow_files.py

Selects a balanced set of real workflow instances from a local checkout of
WfCommons/WfInstances (https://github.com/wfcommons/WfInstances), for use
with pipeline_real_data.py.

Balancing logic:
  - Deduplicates by distinct module count per application, so near-identical
    replicate runs (e.g. srasearch-chameleon-10a-001.json vs -002.json,
    which have the same workflow structure/size) don't crowd out other
    applications.
  - Allocates roughly evenly across applications, capped by how many
    distinct sizes are actually available for each one, topping up from
    the largest available pools when a small pool runs out.
  - Caps included workflows at a maximum module count to keep per-trial
    QPSO/GA runtime tractable (raise MAX_MODULES if you have more compute
    budget and want larger instances included).

Usage:
    python3 select_workflow_files.py \\
        --source_dir wf_raw/WfInstances-main/pegasus \\
        --target_total 60 \\
        --max_modules 700 \\
        --out_filelist real_workflow_filelist_60.txt \\
        --copy_to real_workflows_60
"""

import argparse
import glob
import json
import os
import shutil


def load_module_counts(source_dir, max_modules):
    """Returns {app: [(n_modules, filepath), ...]} deduplicated by distinct
    module count per app, filtered to n_modules <= max_modules."""
    apps = {}
    for f in sorted(glob.glob(os.path.join(source_dir, '*', '*.json'))):
        app = f.split(os.sep)[-2]
        try:
            with open(f) as fh:
                d = json.load(fh)
            n = len(d['workflow']['specification']['tasks'])
        except Exception:
            continue
        if n > max_modules:
            continue
        apps.setdefault(app, {})
        if n not in apps[app]:  # keep first file seen per distinct size
            apps[app][n] = f
    return {app: sorted(sizes.items()) for app, sizes in apps.items()}


def allocate_balanced(apps, target_total):
    """Even allocation per application, capped by availability, topped up
    from apps with remaining capacity (largest pools first) to hit target_total."""
    apps_list = sorted(apps.keys())
    avail = {a: len(apps[a]) for a in apps_list}
    n_apps = len(apps_list)

    alloc = {a: min(avail[a], target_total // n_apps) for a in apps_list}
    remaining = target_total - sum(alloc.values())

    order = sorted(apps_list, key=lambda a: -avail[a])
    i = 0
    while remaining > 0 and i < 10000:
        a = order[i % len(order)]
        if alloc[a] < avail[a]:
            alloc[a] += 1
            remaining -= 1
        i += 1
    return alloc


def select_evenly_spaced(files, k):
    """Pick k items evenly spaced across a sorted (size, path) list."""
    if k >= len(files):
        return files
    if k <= 1:
        return [files[0]]
    idxs = sorted(set(round(x * (len(files) - 1) / (k - 1)) for x in range(k)))
    return [files[idx] for idx in idxs]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', required=True,
                         help='Path to WfInstances pegasus/ directory (from the extracted repo tarball)')
    parser.add_argument('--target_total', type=int, default=60)
    parser.add_argument('--max_modules', type=int, default=700,
                         help='Exclude workflows larger than this many modules (compute-tractability cap)')
    parser.add_argument('--out_filelist', default='real_workflow_filelist.txt')
    parser.add_argument('--copy_to', default=None,
                         help='If given, copy selected files into this directory (organized by app subfolder)')
    args = parser.parse_args()

    apps = load_module_counts(args.source_dir, args.max_modules)
    print('Available distinct-size instances per application:')
    for app, files in sorted(apps.items()):
        print(f'  {app:15s} {len(files):3d} available, sizes {files[0][0]}-{files[-1][0]}')

    alloc = allocate_balanced(apps, args.target_total)
    print('\nFinal allocation:', alloc, ' total =', sum(alloc.values()))

    selected = []
    for app in sorted(apps.keys()):
        chosen = select_evenly_spaced(apps[app], alloc[app])
        for n, f in chosen:
            selected.append(f)
            print(f'  selected: {app:15s} {n:4d} modules  {os.path.basename(f)}')

    with open(args.out_filelist, 'w') as f:
        f.write('\n'.join(selected) + '\n')
    print(f'\nWrote {len(selected)} paths to {args.out_filelist}')

    if args.copy_to:
        portable_lines = []
        for src in selected:
            app = src.split(os.sep)[-2]
            dest_dir = os.path.join(args.copy_to, app)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy(src, dest_dir)
            portable_lines.append(os.path.join(args.copy_to, app, os.path.basename(src)))

        portable_path = args.out_filelist.replace('.txt', '_portable.txt')
        with open(portable_path, 'w') as f:
            f.write('\n'.join(portable_lines) + '\n')
        print(f'Copied {len(selected)} files into {args.copy_to}/')
        print(f'Wrote portable filelist (paths relative to {args.copy_to}/) to {portable_path}')


if __name__ == '__main__':
    main()
