Training Subject Documentation

There are four main ways the Zooniverse platform can choose to operate on training subjects.

1. **_Designator_** performs custom subject selection, serving training subjects to volunteers along side real data with adjustable frequency.

- Parameters to determine behavior are set via `workflow.configuration` variables:
    - `training_set_ids` - identifies training subject sets
    - `training_chances` - array of values used to determine frequency at which training subjects are shown. A value is selected from this array via the index value derived from a user's seen subjects count for the workflow.
    - `training_default_chance` - default chance used in absence of, or beyond array-based values
    - `subject_queue_page_size` - a second-order parameter that determines how often new sets of subjects are requested, impacting whether `training_chances` array is properly used.
        - The default number of subjects the API returns to users for classification at a time is 10
        - A setting of 10 results in the `training_chances` array being largely ignored as it will only be checked every ~8-10 classifications.
        - The use of smaller values (~ 4) is recommended when `training_chances` values change rapidly as a function of user seen subjects count.
        - Take care when setting this value lower than 4 as PFE will request new subjects when less than 2 subjects are in the local subject queue cache.

- Designator workflow parameters can be configured via python client by project owners & collaborators.  Here is a set of example commands, for 25% chance of showing training data for first 100 subjects, then default to 5% afterward:
      workflow_id = <WORKFLOW ID>
      workflow = Workflow.find(workflow_id)
      workflow.configuration['training_set_ids'] = [<SUBJECT SET ID>]
      workflow.configuration['training_chances'] = [0.25] * 100 # your ratios!
      workflow.configuration['training_default_chance'] = 0.05
      workflow.configuration['subject_queue_page_size'] = 4
      workflow.modified_attributes.add('configuration') # workaround for save bug
      workflow.save()

2. **_Panoptes_** removes training subjects from a project's subject counts and completion fractions.

- The result: workflows can reach 100% completion irregardless of the retirement status of training subjects.
- Use case: training subjects will not keep a workflow active (rather than paused) if all non-training, experimental subjects are retired.
- This behavior is enabled via the `training_set_ids` workflow configuration parameter.


3. **_Panoptes_** accepts two choices for retirement criteria
- This setting is configured via the `workflow.retirement['criteria']`
- There are two valid settings;
    - `classification_count` with a sibling key value of `'options': {'count': 15}` for the threshold value, use the workflow lab page to set this value
    - `never_retire`:
        - Training-specific workflows that should remain active indefinitely
        - Workflows where all retirement is handled externally when training subjects need to remain active (so they can be served to other users) while non-training subjects retire via Caesar or client actions.
        - Zoo team members can change this setting per-workflow from the project's admin page
- To set retirement criteria to `never_retire`:
      workflow_id = <WORKFLOW ID>
      workflow = Workflow.find(workflow_id)
      workflow.retirement['criteria'] = 'never_retire'
      workflow.modified_attributes.add('retirement') # workaround for save bug
      workflow.save()

4. **_Caesar_** uses reducer filters and preferential extract creation to treat training subjects differently.
- Training subjects are identified in Caesar via a subject metadata tag, `#training_subject`, which can be set to `true` or `false` but must be `true` for training subjects.
- The `training_behavior` strategy can be used for reducers to filter extracts according to one of three approaches:
    - `ignore_training` (default; include all training and non-training extracts)
    - `training_only` (keep extracts of training subjects only)
    - `experiment_only` (keep extracts of non-training subjects only).
- The TESS project uses a `PluckExtractor` that is configured to pluck a subject metadata field only available on training subjects (here, a feedback parameter). When combined with the use of a `if_missing` criteria set to `reject`, this configuration creates extracts _only_ for training subjects. This setup acts as a method for filtering extract creation based on subject training status.
