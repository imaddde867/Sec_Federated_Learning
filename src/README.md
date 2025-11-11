# Source Structure

## Core-Safe
`core-safe/` holds the stable, reviewed implementation.  
Do not edit files here directly — only update via a pull request after testing in your experiment folder.

## Experiments
`/<name>/` is your personal sandbox, example :  `imad-yan/` or `Celesteritesh` 
You can freely modify, test, and push work there in your experimental folders without affecting the core implementation.
Just make a new one with your name, copy original code from `core-safe/` and experiment freely.

## Workflow
1. Copy files you need from `core/` into your folder under `experiments/`.
2. Develop, experiment and test there.
3. When ready, open a PR from your branch to merge your changes back into `core/`.