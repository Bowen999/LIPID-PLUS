# LIPID+: Lipid Identification and Profiling for Intelligent Discovery +

![LIPID PLUS](https://github.com/Bowen999/lipid-plus-docs/blob/main/images/lipid_plus_pipeline.png)

[**LIPID+**](https://bowen999.github.io/lipid-plus-docs) is a machine learning-powered platform designed to overcome critical bottlenecks in lipidomics. By shifting analytical focus from unattainable complete structures to primary lipid structural features, LIPID+ provides accurate, comprehensive, and scalable lipid identification, specifically targeting the dark lipidome where reference spectra are missing.

**Key Functions & Features**
* **Pure Lipids MS Database**: A large-scale, curated lipid database containing lipids from diverse sources, along with their precursor m/z values and corresponding MS² spectra, enables high-throughput mass spectral searching.  
* **SLAM (Spectrum-based Lipid Annotation Model)**: A cascaded generative ML model that performs de novo annotation, enabling the identification of novel lipids without relying on existing spectral libraries.  
* **Interactive System-level Analysis**: Tailored analytical and visualization tools designed specifically for lipid characteristics allow detailed exploration of lipid-specific structural features—such as chain length and degree of unsaturation—capturing their dynamic changes. The analytical results are presented as interactive and easily shareable HTML reports.  

# Installation
### Clone the project folder and download database 
 
```bash
git clone https://github.com/Bowen999/LIPID-PLUS.git
cd LIPID-PLUS
mkdir -p dataset
wget -O dataset/lipid_plus.db https://github.com/Bowen999/LIPID-PLUS/releases/download/v1.0/lipid_plus.db

# (if no wget，use curl)
# curl -L -o dataset/lipid_plus.db https://github.com/Bowen999/LIPID-PLUS/releases/download/v1.0/lipid_plus.db
```

### Install the dependencies 

```
conda create -n lipid_plus python=3.12 -y
conda activate lipid_plus

pip install -r requirements.txt
```


# Quick Start

### Using the All-in-One Identification Pipeline

The easiest way to run the complete identification pipeline (database search + machine learning prediction) with default settings:

```bash
python run.py feature_df.csv
```

The usage of the pipeline with custom parameters can be found in the `Advanced Usage` section.  

The pipeline will generate several files in the `results/` directory. **Main result file**: `results/final_annotations.csv` contains your complete lipid annotations.

```text
results/
├── identification_result.csv     # Final merged output (Database + ML predictions)
├── process_files/
├──── processed_feature_table.csv   # Normalized and unfolded MS2 data (Step 0)
├──── db_matched_df.csv             # Lipids identified by database search (Step 1)
├──── dark_lipid.csv                # Unknown lipids sent to prediction pipeline
├──── adduct_predictions.csv        # Intermediate adduct predictions (Step 2)
├──── class_predictions.csv         # Intermediate class predictions (Step 3)
└──── final_annotations.csv         # Final ML chain composition predictions (Step 4)
```



### Lipidomics Analysis Report Generation
An interactive, shareable, and fully customized dynamic lipidomics HTML report (see [example](https://bowen999.github.io/lipid-plus-docs/example_report.html)) can be generated from the previous identification results (identification_result.csv). An example is shown below:

```
python code/report_generate.py \
  --input_path results/identification_result.csv \
  --groups 0h 2h 4h 8h \
  --group_1 0h \
  --group_2 8h \
  --p_value_threshold 0.1 \
  --fc_threshold 1
```

To run this process, the input file must contain at least 3 groups along with intensity or concentration values.

| Parameter | Type | Description |
| :--- | :--- | :--- | :--- |
| `--input_path` | `String` | The file path to the identification result|
| `--groups` | `List` | Space-separated list of all group prefixes present in the CSV columns (e.g., `Control Treated`). |
| `--group_1` | `String`| The first group for differential analysis. Must be in `--groups`. |
| `--group_2` | `String` | The second group for differential analysis. Must be in `--groups`. |

# Advance Usage
**For Advance Usage, please look at [Docs](https://bowen999.github.io/lipid-plus-docs/docs.html)**

# Contact
If you have any questions, please reach out to by8@ualberta.ca