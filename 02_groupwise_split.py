import pandas as pd

# 1. Load phylum-level abundance (samples are columns)
phylum_abundance = pd.read_csv("phylum_level_abundance.csv", index_col=0)

# 2. Load sample metadata
metadata = pd.read_csv("sampleID.csv")  # Or read_csv()

# 3. Extract required columns
metadata = metadata.iloc[:, [0, 4, 11]]  # Sample ID, Gender, Disease State
metadata.columns = ['sample.ID', 'Gender', 'Disease']  # Rename for clarity

# 4. Normalize text to lower case and strip spaces
metadata['sample.ID'] = metadata['sample.ID'].astype(str).str.strip()
metadata['Gender'] = metadata['Gender'].astype(str).str.lower().str.strip()
metadata['Disease'] = metadata['Disease'].astype(str).str.strip()

# 5. Prepare output writer
with pd.ExcelWriter("phylum_abundance_by_group.xlsx") as writer:
    for disease in ['Healthy', 'NAFLD']:
        for gender in ['male', 'female']:
            # Filter sample IDs for this group
            group_samples = metadata[
                (metadata['Disease'] == disease) &
                (metadata['Gender'] == gender)
            ]['sample.ID']

            # Keep only those columns (samples) in phylum_abundance
            matching_columns = [s for s in group_samples if s in phylum_abundance.columns]

            # Subset the abundance table
            if matching_columns:
                group_df = phylum_abundance[matching_columns]
                sheet_name = f"{disease}_{gender}"
                group_df.to_excel(writer, sheet_name=sheet_name)
            else:
                print(f"No samples for group {disease}_{gender}")

