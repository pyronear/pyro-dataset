# TODO list

- [ ] Generate reports for the temporal datasets as well.
- [ ] Update README.md with new stages for temporal dataset
- [ ] Improve documentation to explain the process to add new data into the dataset
- [ ] Remove the FP2024 dependencies as it is a messy dataset containing a lot of red bboxes that spreads in our datasets
- [ ] Create a simple UI for quickly iterating over new sequences pulled from the pyronear API
- [x] Improve the fetch sequences script to not overwrite the sequences.csv and api.csv and args.yaml and merge the folders
- [x] Fetch data from the Chile stations
- [x] Find a good process to annotate quickly the smoke sequences that were found and add them back into the raw dataset in `labels_ground_truth`
- [x] Add some quality check tests on the raw data (platform-annotated-sequences to start with)
- [x] Add a new ratio parameter for splitting the sequences into train/val/test (one for smoke and one for backgrounds)
  - We want many smoke sequences in test to have a balanced dataset
  - We do not need that many smoke sequences in train or val
- [x] Include a manifest.yaml for the temporal dataset too
