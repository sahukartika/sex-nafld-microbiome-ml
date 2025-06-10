import pandas as pd

# 1. Load the abundance matrix
abundance_df = pd.read_csv("vect_atlas.csv", index_col=0)  # or pd.read_csv if it's CSV

# 2. Load the mapping file with all sheets
phylum_sheets = pd.read_excel("mspmap.xlsx", sheet_name=None)  # all sheets as dict

# 3. Create a new dataframe to store phylum-level abundances
phylum_abundance = pd.DataFrame(0, index=phylum_sheets.keys(), columns=abundance_df.columns)

# 4. Loop through each phylum and sum the abundance of its MSPs
for phylum, sheet_df in phylum_sheets.items():
    # Normalize MSP IDs in both files
    abundance_df.index = abundance_df.index.astype(str).str.strip().str.lower()
    sheet_df.iloc[:, 0] = sheet_df.iloc[:, 0].astype(str).str.strip().str.lower()

    msp_ids = sheet_df.iloc[:, 0]
    matching_msps = abundance_df.loc[abundance_df.index.isin(msp_ids)]
  # first column = MSP IDs
    
    phylum_abundance.loc[phylum] = matching_msps.sum()

# 5. Save or inspect the phylum-level abundance
phylum_abundance.to_csv("phylum_level_abundance.csv")
print(phylum_abundance.head())
print("Example MSPs in mapping sheet:", msp_ids[:5].tolist())
print("Example MSPs in abundance:", abundance_df.index[:5].tolist())
