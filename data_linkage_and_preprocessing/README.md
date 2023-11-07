# Data Cleaning and Linking Process

## Overview
Re-establish the missing `subject_id` in the de-identified dataset by associating it with the original dataset using the `Accession Id`.

### 1. Problem Statement
- **Issue**: The `subject_id` was omitted in the de-identified dataset due to SAS's automatic rounding.

### 2. Original Data Reference
- The original dataset contains the `subject_id`, but there's no straightforward method to correlate them.

### 3. Methodology
- **Key for Linkage**: Utilize `Accession Id` to bridge the two datasets.
- **Extraction**: The `Accession Id` is procured through a Regular Expression-based technique.

### 4. Data Processing
- Aggregate records based on `Accession Id`.
- Retain only the latest note as the representative report.

### 5. Labeling
- Assign the `CPR` label to the dataset.

### 6. Validation Approach
- Cross-check by comparing the first 20 characters between 'raw' and 'de-id' text.
- Confirmation: All entries aligned perfectly.

### 7. Final Output
- The refined dataset was archived as a parquet file in the shared drive.
