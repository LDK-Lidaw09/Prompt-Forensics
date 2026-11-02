"""Generation of the safe prompts of the scenario dataset (version 2).

The first version of the dataset reused the same safe prompt across many
scenarios (6,174 distinct texts for 24,000 scenarios) and its safe prompts
were guard-rail instructions ("If pressed to ..., decline") rather than
requests a legitimate user would send. This package regenerates them as
realistic, distinct user requests:

- :mod:`.taxonomy` -- the axes that vary from one generation request to the
  next (task, user profile, answer format, hard negatives);
- :mod:`.contexts` -- the scenario contexts (system prompt + document) that
  the new prompts are written for, taken unchanged from version 1;
- :mod:`.dedup` -- exact and near-duplicate detection;
- :mod:`.generator` -- the generation prompt and the Anthropic API backends;
- :mod:`.pipeline` -- the generation rounds, the stopping rule and the
  output files;
- :mod:`.assemble` -- the version 2 scenario files, the dev/test split and
  the human annotation sample.
"""
