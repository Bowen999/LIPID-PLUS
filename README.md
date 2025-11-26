# Installation
**Run the following code from the command line.**

Clone the project folder 
 
```bash
git clone https://github.com/pluskal-lab/DreaMS.git
cd DreaMS
```

Download DB

```
```

Install the dependencies 

```
conda create -n lipid_plus python=3.12 -y
conda activate lipid_plus

pip install -r requirements.txt
```




### Directory Structure

For the simplest usage, organize your files as follows:

```
your_project/
├── run.py                          # The main pipeline script
├── model/                          # Model directory
│   ├── adduct.joblib              # Adduct prediction model
│   ├── class.joblib               # Class prediction model
│   └── plsf.joblib                # PLSF model
├── dataset/                        # Database directory
│   └── lipid.db                   # Reference database
├── code/                           # All step scripts
│   ├── db_search.py               # Step 1: Database search
│   ├── adduct_predict.py          # Step 2: Adduct prediction
│   ├── class_predict.py           # Step 3: Class prediction
│   ├── predict_plsf.py            # Step 4: PLSF prediction
│   └── formula_annotation.py      # Alternative workflow
├── feature_df.csv                  # Your input data
└── results/                        # Output directory (created automatically)
```

With this structure, you can simply run:
```bash
python run.py feature_df.csv
```

### File Requirements

Before running the pipeline, ensure you have:

1. **Input data** (CSV file with MS/MS spectra)
2. **Trained models** in `model/` directory:
   - `adduct.joblib` - Adduct prediction model
   - `class.joblib` - Class prediction model
   - `plsf.joblib` - Chain composition prediction model
3. **Database** (optional) in `dataset/` directory:
   - `lipid.db` or `lipid_plus.db` - SQLite database with reference lipids

---

# Quick Start

### Using the All-in-One Identification Pipeline

The easiest way to run the complete annotation pipeline with default settings:

```bash
python run.py feature_df.csv
```

The pipeline will use default paths:
- **Models**: `model/adduct.joblib`, `model/class.joblib`, `model/plsf.joblib`
- **Database**: `dataset/lipid.db`
- **Results**: `results/`


### Customizing Parameters (Optional)

If you need different settings:

```bash
# Custom result directory
python run.py feature_df.csv --result_path my_results/

# Different MS tolerance
python run.py feature_df.csv \
    --MS1_tol 0.01 \
    --MS2_threshold 0.8
```

### Expected Output

After running:
```bash
python run.py feature_df.csv
```

The pipeline will generate several files in the `results/` directory:

```
results/
├── annotated_df.csv              # Database-matched lipids
├── dark_lipid.csv                # Unknown lipids for prediction
├── adduct_predictions.csv        # Step 2 output
├── class_predictions.csv         # Step 3 output
├── final_annotations.csv         # Step 4 output (MAIN RESULT)
└── all_annotations_combined.csv  # Combined database + predicted
```

**Main result file**: `results/final_annotations.csv` contains your complete lipid annotations!

### Report Generate

```
```

---

# Advanced Usage
## Database Search

Searches a spectral database to identify known lipids based on precursor m/z and MS/MS similarity.

#### Command

```bash
python db_search.py input.csv \
    --result_path results/ \
    --db_path lipid.db \
    --MS1_tol 0.005 \
    --MS2_tol 0.01 \
    --MS2_threshold 0.7 \
    --method weighted_dot_product \
    --is_ppm False
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `csv_path` | str | *required* | Path to input CSV file |
| `--result_path` | str | `.` | Output directory |
| `--db_path` | str | `lipid_plus.db` | Path to SQLite database |
| `--MS1_tol` | float | `0.005` | MS1 tolerance (Da or ppm) |
| `--MS2_tol` | float | `0.01` | MS2 fragment tolerance (Da) |
| `--method` | str | `weighted_dot_product` | Scoring method: `dot_product`, `weighted_dot_product`, `entropy_similarity`, `unweighted_entropy_similarity` |
| `--MS2_threshold` | float | `0.7` | Minimum MS2 score (0-1) to accept a match |
| `--is_ppm` | bool | `False` | Use ppm for MS1 tolerance (True) or Da (False) |

#### Input Requirements

Your CSV must contain:
- `precursor_mz`: Precursor mass-to-charge ratio
- `adduct`: Adduct type (e.g., `[M+H]+`, `[M+Na]+`, `[M-H]-`)
- `MS2`: MS/MS spectrum as string representation of list, e.g., `"[[100.0, 1500], [200.5, 3000]]"`

#### Output Files

1. **`annotated_df.csv`**: Successfully matched lipids with database annotations
   - Contains all original columns plus:
   - `name`, `formula`, `class`, `category` (from database)
   - `mass_diff_ppm`: Mass accuracy
   - Spectral similarity scores
   
2. **`dark_lipid.csv`**: Unknown lipids that didn't match the database
   - These will be processed by the prediction steps

---

## Adduct Prediction

Predicts the adduct type for unknown lipids using a trained machine learning model.

#### Command

```bash
python adduct_predict.py input.csv model/adduct.joblib \
    --output_path results/adduct_predictions.csv
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_path` | str | *required* | Input CSV file |
| `model_path` | str | *required* | Path to trained adduct model (.joblib) |
| `--output_path` `-o` | str | `adduct_predict.csv` | Output file path |

#### Input Requirements

The CSV must contain:
- `precursor_mz`: Precursor m/z (will be rounded to 2 decimals)
- `ion_mode`: Ion mode (`positive` or `negative`)
- `mz_*` columns: Binary features (0/1) indicating presence of fragment ions

#### Output Columns

Original columns plus:
- `predicted_adduct`: Predicted adduct type
- `adduct_confidence`: Prediction confidence (0-1), if available

#### Example

```bash
# Using dark lipids from Step 1
python adduct_predict.py results/dark_lipid.csv models/adduct.joblib \
    -o results/with_adducts.csv

# Process all data (no database search)
python adduct_predict.py raw_data.csv adduct.joblib \
    -o predictions/adducts.csv
```

---

## Class Prediction

Predicts lipid class using a hybrid approach combining rule-based classification and machine learning.

#### Command

```bash
python class_predict.py input.csv model/class.joblib \
    --output_path results/class_predictions.csv \
    --ms1_tol 10 \
    --ms2_tol 20
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_path` | str | *required* | Input CSV file |
| `model_path` | str | *required* | Path to trained class model (.joblib) |
| `--output_path` `-o` | str | `class_pred.csv` | Output file path |
| `--ms1_tol` | float | `10.0` | MS1 tolerance for rule-based matching (ppm) |
| `--ms2_tol` | float | `20.0` | MS2 tolerance for rule-based matching (ppm) |
| `--no-rules` | flag | `False` | Skip rule-based classification, use ML only |

#### Input Requirements

The CSV must contain:
- `precursor_mz`: Precursor m/z
- `ion_mode`: Ion mode
- `adduct` or `predicted_adduct`: Adduct type
- `mz_*` columns: Fragment features
- `MS2` or `MS2_norm`: MS/MS spectrum (for rule-based classification)

#### Two-Stage Process

1. **Rule-Based Classification**: Uses chemical knowledge and fragmentation patterns
   - High confidence for lipids with characteristic fragments
   - Marks successfully classified lipids with `prediction_source = 'rule-based'`

2. **ML-Based Prediction**: For remaining lipids
   - Uses machine learning model
   - Marks predictions with `prediction_source = 'model-based'`

#### Output Columns

Original columns plus:
- `predicted_class`: Predicted lipid class (e.g., PC, PE, TG, Cer)
- `prediction_source`: `'rule-based'` or `'model-based'`
- `class_confidence`: Confidence score (0-1)
- `category`: Lipid category (e.g., Glycerophospholipids, Sphingolipids)
- `num_chain`: Number of fatty acid chains
- `classes_mz`: Possible classes from precursor mass (rule-based)
- `classes_ms2`: Possible classes from MS2 fragments (rule-based)

#### Example

```bash
# Default (hybrid approach)
python class_predict.py results/with_adducts.csv class.joblib \
    -o results/with_classes.csv

# ML only (faster, may be less accurate)
python class_predict.py input.csv class.joblib \
    -o output.csv --no-rules

# Strict tolerances (high-resolution data)
python class_predict.py input.csv class.joblib \
    -o output.csv --ms1_tol 5 --ms2_tol 10
```

---

## Chain Composition Prediction

Predicts detailed fatty acid chain compositions (number of carbons and double bonds in each chain).

#### Command

```bash
python predict_plsf.py model/plsf.joblib input.csv \
    --output_path results/final_output.csv
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | str | *required* | Path to trained PLSF model (.joblib) |
| `input_path` | str | *required* | Input CSV file |
| `-o` `--output_path` | str | `result.csv` | Output file path |

#### Input Requirements

The CSV must contain:
- `precursor_mz`: Precursor m/z (rounded to 2 decimals)
- `adduct` or `predicted_adduct`: Adduct type
- `class` or `predicted_class`: Lipid class
- `num_chain`: Number of fatty acid chains (1-4)
- `mz_*` columns: Fragment features

#### Special Handling

- **Single-chain lipids** (`num_chain=1`): Direct mass calculation instead of model
- **Unknown adducts/classes**: Marked as invalid, predictions set to NaN
- **Confidence scores**: Combined from PLSF, class, and adduct predictions

#### Output Columns

Cleaned output with:
- `name`: Combined lipid name (e.g., "PC 16:0_18:1")
- `precursor_mz`, `ion_mode`, `adduct`, `class`, `category`
- `num_chain`: Number of chains
- `pred_confidence`: Average confidence across all prediction steps
- `plsf_rank1`, `plsf_rank2`, `plsf_rank3`: Top 3 chain composition predictions
- (mz_* features and intermediate prediction columns are removed)

#### Name Format

The `name` column combines class and chain information:
- Format: `{Class} {C1}:{DB1}_{C2}:{DB2}_{C3}:{DB3}_{C4}:{DB4}`
- Chains sorted by length (descending)
- Zero chains (0:0) are omitted
- Examples:
  - `PC 16:0_18:1` (PC with C16:0 and C18:1)
  - `TG 16:0_18:1_18:2` (TG with three chains)
  - `Cer 18:1_16:0` (Ceramide with two chains)

#### Example

```bash
# Basic usage
python predict_plsf.py plsf.joblib results/with_classes.csv \
    -o results/final_annotations.csv

# With custom output location
python predict_plsf.py models/plsf_split.joblib data/classified.csv \
    -o output/experiment1/annotations.csv
```

---

## Alternative: Formula Annotation

For metabolomics or when molecular formulas are needed, use the formula annotation workflow instead of or in addition to the main pipeline.

#### Command

```bash
python formula_annotation.py input.csv \
    --output_path results/formulas/ \
    --ms1_tol 15 \
    --ms2_tol 20
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_path` | str | *required* | Input CSV file |
| `--output_path` | str | `result/process_file` | Output directory |
| `--ms1_tol` | float | `15.0` | MS1 tolerance for formula annotation (ppm) |
| `--ms2_tol` | float | `20.0` | MS2 tolerance for formula annotation (ppm) |

#### What It Does

1. Converts CSV to MGF format
2. Uses MsBuddy to predict molecular formulas
3. Applies class-specific rules for lipid formulas
4. Generates top 5 formula candidates

#### Output Files

- `input.mgf`: MGF format for MsBuddy
- `buddy_result.csv`: Raw MsBuddy predictions
- `formula_result.csv`: Processed formulas with class rules applied

#### Output Columns

- `predicted_formula`: Top-ranked molecular formula
- `formula_rank_1` to `formula_rank_5`: Top 5 formula candidates
- All original input columns

#### Example

```bash
# High-resolution data (tight tolerances)
python formula_annotation.py data.csv \
    --output_path results/formulas/ \
    --ms1_tol 10 \
    --ms2_tol 15

# Lower-resolution data
python formula_annotation.py data.csv \
    --output_path results/ \
    --ms1_tol 20 \
    --ms2_tol 25
```

---
# FAQ

## Input Data Format

### Required Columns

Your input CSV file must contain these columns:

```csv
index,precursor_mz,ion_mode,adduct,MS2,mz_100,mz_101,...
F000001,760.5851,positive,[M+H]+,"[[184.07, 5000], [104.11, 2000]]",1,0,...
F000002,788.6164,positive,[M+H]+,"[[184.07, 8000], [506.36, 1500]]",1,0,...
```

#### Column Descriptions

| Column | Type | Description | Required For |
|--------|------|-------------|--------------|
| `index` | str | Unique identifier | All steps |
| `precursor_mz` | float | Precursor mass-to-charge ratio | All steps |
| `ion_mode` | str | `positive` or `negative` | Adduct prediction |
| `adduct` | str | Adduct type (e.g., `[M+H]+`) | DB search, class prediction |
| `MS2` | str/list | MS/MS spectrum as `[[mz, intensity], ...]` | DB search, class prediction |
| `mz_*` | int | Binary features (0/1) for fragment presence | All predictions |

### MS2 Spectrum Format

The `MS2` column can be formatted as:

```python
# String representation (preferred for CSV)
"[[184.07, 5000], [104.11, 2000], [506.36, 1500]]"

# Or after loading, as a Python list
[[184.07, 5000], [104.11, 2000], [506.36, 1500]]
```

### Feature Columns (mz_*)

- Binary features indicating presence/absence of fragment ions
- Typically generated during preprocessing
- Example: `mz_184`, `mz_104`, `mz_86`, etc.
- Values: `0` (absent) or `1` (present)

---

## Output Files

### Final Annotations (`final_annotations.csv`)

The main output containing complete lipid annotations:

```csv
index,name,precursor_mz,ion_mode,adduct,class,category,num_chain,pred_confidence,plsf_rank1,...
F000001,PC 16:0_18:1,760.5851,positive,[M+H]+,PC,Glycerophospholipids,2,0.94,"16:0_18:1",...
```

### Database Annotations (`annotated_df.csv`)

Lipids successfully matched to the database:

```csv
index,precursor_mz,adduct,name,formula,class,category,mass_diff_ppm,weighted_dot_product,...
F000015,760.5851,[M+H]+,PC(16:0/18:1),C42H82NO8P,PC,Glycerophospholipids,-0.8,0.89,...
```

### Combined Results (`all_annotations_combined.csv`)

Merged database annotations + predicted annotations for comprehensive results.

---

## Troubleshooting

### Common Issues

#### 1. Missing Columns Error

**Error:** `✗ Missing required columns: ['mz_184', 'mz_104']`

**Solution:** Ensure your input CSV has all required `mz_*` feature columns. These are typically generated during data preprocessing.

#### 2. Model Loading Error

**Error:** `✗ Error loading model: No such file or directory`

**Solution:** Verify the model path is correct and the .joblib file exists:
```bash
ls -la models/
```

#### 3. Unknown Adduct/Class Values

**Warning:** `⚠ Warning: Found 15 unseen adduct(s): {'[M+2H]2+'}`

**Solution:** The model hasn't seen this adduct during training. Predictions for these samples will be marked as "unknown". Consider:
- Using only known adducts
- Retraining the model with additional adduct types

#### 4. Low Confidence Predictions

**Issue:** Many predictions have confidence < 0.5

**Solutions:**
- Check data quality (MS2 spectra quality, peak picking)
- Verify preprocessing (correct m/z binning, feature extraction)
- Review tolerances (may be too strict or too loose)
- Consider rule-based classification for better accuracy

#### 5. Database Connection Error

**Error:** `Error: Database file 'lipid.db' not found!`

**Solution:** Either:
- Provide correct path to database: `--db_path /path/to/lipid.db`
- Skip database search and use prediction-only mode

#### 6. Empty Output Files

**Issue:** Output CSV has 0 rows

**Solutions:**
- Check if input data has required columns
- Verify MS2 spectra are in correct format
- Review filter thresholds (MS2_threshold, tolerances)
- Check for data type mismatches (e.g., string vs. list for MS2)

---

## Tips for Best Results

### 1. Data Preprocessing
- **Quality control**: Remove low-quality spectra (too few peaks, low intensity)
- **Peak picking**: Use consistent parameters across samples
- **Feature extraction**: Ensure mz_* features are properly generated

### 2. Parameter Tuning
- **High-resolution data**: Use tighter tolerances (MS1: 5-10 ppm, MS2: 0.01 Da)
- **Low-resolution data**: Use looser tolerances (MS1: 20 ppm, MS2: 0.05 Da)
- **MS2_threshold**: Start with 0.7, increase for stricter matching

### 3. Model Selection
- **Hybrid approach**: Use rule-based + ML for best accuracy (default)
- **ML only**: Use `--no-rules` flag for faster processing
- **Rule-based only**: Stop after class_predict.py with `--no-ml` (if implemented)

### 4. Validation
- **Manual inspection**: Review a sample of predictions
- **Confidence scores**: Filter results by `pred_confidence > 0.7`
- **Cross-reference**: Compare with database annotations when available

### 5. Batch Processing
```bash
# Process multiple files
for file in data/*.csv; do
    python run.py "$file" \
        --adduct_model models/adduct.joblib \
        --class_model models/class.joblib \
        --plsf_model models/plsf.joblib \
        --result_path "results/$(basename $file .csv)/"
done
```

---

## Citation

If you use this pipeline in your research, please cite:
```
[Your citation information here]
```

---

## Support

For issues, questions, or contributions:
- GitHub: [Your repository]
- Email: [Your contact email]
- Documentation: [Your documentation site]

---

**Last Updated:** November 2024  
**Version:** 1.0.0
