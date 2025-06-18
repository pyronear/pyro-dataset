# TODO list

- [ ] Remove the FP2024 dependencies as it is a messy dataset containing a lot of red bboxes that spreads in our datasets
- [ ] Improve the fetch sequences script to not overwrite the sequences.csv and api.csv and args.yaml and merge the folders
- [ ] Fetch data from the Chile stations
- [ ] Find a good process to annotate quickly the smoke sequences that were found and add them back into the raw dataset in `labels_ground_truth`
- [ ] Add some quality check tests on the raw data (platform-annotated-sequences to start with)
- [ ] Create a simple UI for quickly iterating over new sequences pulled from the pyronear API
- [ ] Update README.md with new stages for temporal dataset
- [ ] Improve documentation to explain the process to add new data into the dataset
