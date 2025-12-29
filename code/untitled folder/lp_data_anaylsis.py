# Import necessary libraries
import pandas as pd
import numpy as np
import re
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import pandas as pd
import numpy as np
import json
from IPython.display import display, HTML
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import pandas as pd
import json
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.colors

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.colors
import os
import pandas as pd
import plotly.express as px
import os
import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import ThemeRiver
from pyecharts.commons.utils import JsCode
import os

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import plotly.express as px
import plotly.graph_objects as go

import pandas as pd
import numpy as np
import json




def merge_and_average_columns(df: pd.DataFrame, col_prefixes: list) -> pd.DataFrame:
    """
    Merges DataFrame columns based on a list of prefixes and calculates their average.

    For each prefix in the col_prefixes list, this function finds all columns
    in the DataFrame that contain that prefix. It then calculates the row-wise
    average of these columns and creates a new column named after the prefix.
    The original columns that were merged are dropped.

    Args:
        df: The input pandas DataFrame.
        col_prefixes: A list of strings, where each string is a prefix to
                      identify columns to be merged.

    Returns:
        A new pandas DataFrame with the specified columns merged and averaged.
    """
    # Create a copy to avoid modifying the original DataFrame
    df_copy = df.copy()
    
    # Keep track of all columns that are merged to drop them later
    cols_to_drop = set()

    # Iterate over each prefix in the provided list
    for prefix in col_prefixes:
        # Find all columns in the DataFrame that contain the current prefix
        cols_to_merge = [col for col in df_copy.columns if prefix in col]
        
        # If any columns with the prefix are found
        if cols_to_merge:
            # Select the columns to be merged
            subset_df = df_copy[cols_to_merge]
            
            # Ensure all data in the subset is numeric, coercing non-numeric to NaN
            numeric_subset = subset_df.apply(pd.to_numeric, errors='coerce')
            
            # Calculate the mean for each row, ignoring NaN values
            # The new column will be named after the prefix
            df_copy[prefix] = numeric_subset.mean(axis=1)
            
            # Add the merged columns to our set of columns to drop
            cols_to_drop.update(cols_to_merge)
        else:
            # If no columns are found for a prefix, you might want to handle it
            # For example, create a column of NaNs or print a warning
            print(f"Warning: No columns found containing the prefix '{prefix}'.")
            df_copy[prefix] = np.nan

    # Drop the original columns that have been merged and averaged
    df_final = df_copy.drop(columns=list(cols_to_drop))
    
    return df_final



def analyze_differential_expression(df, group1_id, group2_id, fc_threshold=2.0, p_value_threshold=0.05):
    """
    Performs differential expression analysis between two groups in a DataFrame.

    This function calculates the log2 fold change and p-value for each row (gene/protein)
    by comparing two specified sample groups. It then filters the results based on
    user-defined thresholds.

    Args:
        df (pd.DataFrame): The input DataFrame. Rows represent features (e.g., genes),
                           and columns represent samples.
        group1_id (str): A string identifier unique to the column names of the first group
                         (e.g., '0h', 'control').
        group2_id (str): A string identifier unique to the column names of the second group
                         (e.g., '8h', 'treatment').
        fc_threshold (float, optional): The absolute fold change threshold for filtering.
                                      Defaults to 2.0. Note this is not log2 transformed.
        p_value_threshold (float, optional): The p-value significance threshold.
                                            Defaults to 0.05.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two DataFrames:
            - df_significant: A DataFrame filtered for significant results.
            - df_all_stats: A DataFrame with all original rows and added stats columns.
    """
    # --- 1. Identify columns for each group ---
    group1_cols = [col for col in df.columns if group1_id in col]
    group2_cols = [col for col in df.columns if group2_id in col]

    if not group1_cols or not group2_cols:
        raise ValueError("Could not find columns for one or both group identifiers."
                         f" Group 1 ('{group1_id}') found: {len(group1_cols)} columns."
                         f" Group 2 ('{group2_id}') found: {len(group2_cols)} columns.")

    print(f"Found {len(group1_cols)} columns for Group 1 ('{group1_id}').")
    print(f"Found {len(group2_cols)} columns for Group 2 ('{group2_id}').")

    # A small constant to avoid division by zero or log(0) errors
    epsilon = 1e-9

    # --- Create a copy to store all results ---
    # CHANGED: Renamed for clarity
    df_all_stats = df.copy()

    # --- 2. Calculate group means ---
    group1_mean = df[group1_cols].mean(axis=1)
    group2_mean = df[group2_cols].mean(axis=1)

    # --- 3. Calculate Fold Change and Log2 Fold Change ---
    df_all_stats['fold_change'] = (group2_mean + epsilon) / (group1_mean + epsilon)
    df_all_stats['log2_fold_change'] = np.log2(df_all_stats['fold_change'])

    # --- 4. Perform Statistical Test (Welch's t-test) ---
    t_stat, p_value = ttest_ind(df[group2_cols], df[group1_cols], axis=1, equal_var=False, nan_policy='omit')
    df_all_stats['p_value'] = p_value
    df_all_stats['-log10_p_value'] = -np.log10(df_all_stats['p_value'] + epsilon)

    # --- 5. Filter based on thresholds ---
    is_significant_p = df_all_stats['p_value'] < p_value_threshold
    is_significant_fc = abs(df_all_stats['fold_change']) > fc_threshold

    # CHANGED: Renamed for clarity
    df_significant = df_all_stats[is_significant_p & is_significant_fc]

    # CHANGED: Return both DataFrames
    return df_significant, df_all_stats




def create_interactive_spectrogram_table(df: pd.DataFrame, file_path: str = "result/report/MS.html"):
    """
    Converts a pandas DataFrame into an advanced interactive HTML table and saves it to a file.

    V11 Changes:
    - Removed the 'Beta Version Notice' banner.
    - Fixed a Python NameError caused by a conflict between f-string
      and JavaScript template literal syntax for 'pageCount'.
    - Updated pagination to use Previous/Next buttons and a page input field
      for better navigation with a large number of pages.
    - Function signature changed to accept a DataFrame and an output file path.
    - Saves the HTML content to the specified path instead of displaying it.
    - Removed the 'correct' column and associated checkbox functionality.
    - Replaces the 'x' marker on the chart with the actual m/z value
      displayed as text on top of each peak.
    - Implements a custom Chart.js plugin to handle the label drawing.

    Args:
        df (pd.DataFrame): The input DataFrame.
        file_path (str): The path to save the generated HTML file.
                         Defaults to "result/report/MS.html".
    """
    df_copy = df.copy()

    # Ensure spectrum data is in a JSON string format for embedding
    for col in ['MS2', 'MS2_norm']:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].apply(
                lambda x: json.dumps(x) if not isinstance(x, str)
                else json.dumps(json.loads(x.replace("'", '"')))
            )

    # Add a unique index to each row for tracking state
    data_with_index = json.loads(df_copy.to_json(orient='records'))
    for i, row in enumerate(data_with_index):
        row['originalIndex'] = i
    data_json = json.dumps(data_with_index)

    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Interactive DataFrame Viewer</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                background-color: #f9fafb;
                color: #333;
            }}
            .container {{ max-width: 1200px; margin: 20px auto; padding: 20px; }}
            .table-wrapper {{ background-color: #ffffff; border-radius: 10px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06); overflow: hidden; }}
            table {{ width: 100%; border-collapse: collapse; table-layout: fixed; }}
            th, td {{ 
                padding: 12px 15px; 
                text-align: left; 
                white-space: nowrap; 
                overflow: hidden; 
                text-overflow: ellipsis; 
            }}
            thead th {{ 
                background-color: #10B981; 
                color: white; 
                font-weight: 600; 
                font-size: 14px; 
                cursor: pointer; 
                position: relative; 
                transition: background-color 0.2s ease-in-out;
            }}
            thead th:hover {{
                background-color: #059669;
            }}
            thead th .sort-indicator {{
                position: absolute;
                right: 10px;
                top: 50%;
                transform: translateY(-50%);
                font-size: 14px;
                opacity: 0.8;
            }}
            tbody tr {{ border-bottom: 1px solid #e5e7eb; }}
            tbody tr:nth-child(even) {{ background-color: #f9fafb; }}
            tbody tr:last-child {{ border-bottom: none; }}
            tbody tr:hover {{ background-color: #f3f4f6; }}
            td {{ font-size: 14px; }}
            .pagination-controls {{
                margin-top: 20px;
                text-align: center;
                display: flex;
                justify-content: center;
                align-items: center;
                gap: 8px;
            }}
            .pagination-controls button {{ background-color: #ffffff; color: #374151; border: 1px solid #d1d5db; padding: 8px 16px; font-size: 14px; margin: 0 4px; cursor: pointer; border-radius: 6px; transition: background-color 0.2s; }}
            .pagination-controls button:hover {{ background-color: #f3f4f6; }}
            .pagination-controls button:disabled {{ cursor: not-allowed; opacity: 0.5; }}
            .pagination-controls input {{
                width: 60px;
                padding: 8px;
                border: 1px solid #d1d5db;
                border-radius: 6px;
                text-align: center;
                font-size: 14px;
            }}
             .pagination-controls span {{
                font-size: 14px;
                color: #4b5563;
                margin: 0 8px;
            }}
            .clickable-link {{ color: #059669; text-decoration: none; font-weight: 500; cursor: pointer; }}
            .clickable-link:hover {{ text-decoration: underline; }}
            .modal {{ display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; overflow: auto; background-color: rgba(0,0,0,0.5); }}
            .modal-content {{ background-color: #fefefe; margin: 10% auto; padding: 20px; border: 1px solid #888; width: 80%; max-width: 800px; border-radius: 8px; position: relative; }}
            .close-button {{ color: #aaa; float: right; font-size: 28px; font-weight: bold; position: absolute; top: 10px; right: 20px; cursor: pointer; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="table-wrapper"><table id="interactive-table"><thead></thead><tbody></tbody></table></div>
            <div class="pagination-controls" id="pagination-controls"></div>
        </div>
        <div id="spectrumModal" class="modal">
            <div class="modal-content">
                <span class="close-button">&times;</span><h3>MS/MS Spectrum</h3>
                <canvas id="modalChart" width="400" height="200"></canvas>
            </div>
        </div>
        <script>
            let data = {data_json};
            const columns = {list(df_copy.columns)};
            const rowsPerPage = 20;
            let currentPage = 1;
            let chartInstance = null;
            let sortState = {{ column: null, direction: 'asc' }};
            let pageCount = 1;

            const modal = document.getElementById('spectrumModal');
            const closeBtn = document.querySelector('.close-button');

            // --- Modal & Chart Functions ---
            function showModal(spectrumDataStr, columnName) {{
                renderChart(spectrumDataStr, columnName);
                modal.style.display = 'block';
            }}
            function closeModal() {{ modal.style.display = 'none'; }}
            closeBtn.onclick = closeModal;
            window.onclick = function(event) {{ if (event.target == modal) closeModal(); }};

            const topLabelsPlugin = {{
                id: 'topLabels',
                afterDatasetsDraw(chart, args, pluginOptions) {{
                    const {{ ctx }} = chart;
                    ctx.save();
                    ctx.font = 'bold 10px Arial';
                    ctx.fillStyle = 'rgba(0, 0, 0, 0.9)';
                    ctx.textAlign = 'center';

                    const datasetMeta = chart.getDatasetMeta(0);
                    datasetMeta.data.forEach((datapoint, index) => {{
                        const {{ x, y }} = datapoint.getProps(['x', 'y'], true);
                        const value = chart.data.datasets[0].data[index];
                        const xValue = value.x; // The m/z value
                        
                        const yPos = y - 5; 

                        if (value.y > 0) {{
                            ctx.fillText(xValue.toFixed(2), x, yPos);
                        }}
                    }});
                    ctx.restore();
                }}
            }};

            function renderChart(modalContent, columnName) {{
                const ctx = document.getElementById('modalChart').getContext('2d');
                const inputData = JSON.parse(modalContent);
                if (!Array.isArray(inputData) || inputData.length === 0) return;

                const chartData = inputData.map(item => ({{ x: item[0], y: item[1] }}));
                const xValues = inputData.map(item => item[0]);
                const yValues = inputData.map(item => item[1]);

                const minX = xValues.length > 0 ? Math.min(...xValues) - 50 : 0;
                const maxX = xValues.length > 0 ? Math.max(...xValues) + 50 : 100;
                
                let maxY = (columnName === 'MS2_norm') ? 110 : (yValues.length > 0 ? Math.ceil(Math.max(...yValues) * 1.1) : 110);

                if (chartInstance) chartInstance.destroy();
                
                chartInstance = new Chart(ctx, {{
                    type: 'bar',
                    data: {{
                        datasets: [
                            {{
                                label: 'Intensity',
                                data: chartData,
                                backgroundColor: '#7BBA8D',
                                barThickness: 3,
                            }}
                        ]
                    }},
                    plugins: [topLabelsPlugin],
                    options: {{
                        scales: {{
                            x: {{ type: 'linear', min: minX, max: maxX, title: {{ display: true, text: 'm/z' }} }},
                            y: {{ min: 0, max: maxY, title: {{ display: true, text: 'Intensity' }} }}
                        }},
                        plugins: {{ 
                            legend: {{ 
                                display: false
                            }},
                            tooltip: {{
                                enabled: true
                            }}
                        }}
                    }}
                }});
            }}

            // --- Sorting Function ---
            function sortData(column) {{
                sortState.direction = (sortState.column === column && sortState.direction === 'asc') ? 'desc' : 'asc';
                sortState.column = column;
                data.sort((a, b) => {{
                    const valA = a[column], valB = b[column];
                    if (valA < valB) return sortState.direction === 'asc' ? -1 : 1;
                    if (valA > valB) return sortState.direction === 'asc' ? 1 : -1;
                    return 0;
                }});
                displayTable(1);
                updateHeaders();
            }}

            function updateHeaders() {{
                document.querySelectorAll('#interactive-table thead th').forEach(th => {{
                    const indicator = th.querySelector('.sort-indicator');
                    if (indicator) {{
                        indicator.textContent = (th.dataset.column === sortState.column) ? (sortState.direction === 'asc' ? '▲' : '▼') : '';
                    }}
                }});
            }}

            // --- Table and Pagination Rendering ---
            function displayTable(page) {{
                currentPage = page;
                const tableBody = document.querySelector('#interactive-table tbody');
                tableBody.innerHTML = '';
                const paginatedItems = data.slice((page - 1) * rowsPerPage, page * rowsPerPage);

                paginatedItems.forEach(item => {{
                    const row = document.createElement('tr');
                    columns.forEach(col => {{
                        const cell = document.createElement('td');
                        if (['MS2', 'MS2_norm'].includes(col) && item[col]) {{
                            const link = document.createElement('a');
                            link.textContent = 'View Spectrum';
                            link.className = 'clickable-link';
                            link.addEventListener('click', () => showModal(item[col], col));
                            cell.appendChild(link);
                        }} else {{
                            cell.textContent = item[col];
                            cell.title = item[col];
                        }}
                        row.appendChild(cell);
                    }});
                    tableBody.appendChild(row);
                }});
                updatePaginationControls();
            }}

            function setupPagination() {{
                const paginationControls = document.getElementById('pagination-controls');
                pageCount = Math.ceil(data.length / rowsPerPage);

                paginationControls.innerHTML = `
                    <button id="prev-page" title="Previous Page">&laquo; Prev</button>
                    <span id="page-info"></span>
                    <button id="next-page" title="Next Page">Next &raquo;</button>
                    <input type="number" id="page-input" min="1" max="${{pageCount}}" style="width: 60px;">
                    <button id="go-to-page">Go</button>
                `;

                document.getElementById('prev-page').addEventListener('click', () => {{
                    if (currentPage > 1) {{
                        displayTable(currentPage - 1);
                    }}
                }});

                document.getElementById('next-page').addEventListener('click', () => {{
                    if (currentPage < pageCount) {{
                        displayTable(currentPage + 1);
                    }}
                }});

                const goToPage = () => {{
                    const pageInput = document.getElementById('page-input');
                    let pageNum = parseInt(pageInput.value, 10);
                    if (!isNaN(pageNum) && pageNum >= 1 && pageNum <= pageCount) {{
                        displayTable(pageNum);
                    }} else {{
                        pageInput.value = currentPage; // Reset if invalid
                    }}
                }};

                document.getElementById('go-to-page').addEventListener('click', goToPage);
                document.getElementById('page-input').addEventListener('keydown', (e) => {{
                    if (e.key === 'Enter') {{
                        goToPage();
                    }}
                }});
            }}

            function updatePaginationControls() {{
                const paginationControls = document.getElementById('pagination-controls');

                if (pageCount <= 1) {{
                    paginationControls.style.display = 'none';
                    return;
                }}
                
                paginationControls.style.display = 'flex';
                document.getElementById('page-info').textContent = `Page ${{currentPage}} of ${{pageCount}}`;
                document.getElementById('page-input').value = currentPage;
                document.getElementById('prev-page').disabled = currentPage === 1;
                document.getElementById('next-page').disabled = currentPage === pageCount;
            }}
            
            function initialize() {{
                const tableHead = document.querySelector('#interactive-table thead');
                const headerRow = document.createElement('tr');
                columns.forEach(col => {{
                    const th = document.createElement('th');
                    th.textContent = col;
                    th.dataset.column = col;
                    th.innerHTML += `<span class="sort-indicator"></span>`;
                    th.addEventListener('click', () => sortData(col));
                    headerRow.appendChild(th);
                }});
                tableHead.appendChild(headerRow);
                setupPagination();
                displayTable(1);
            }}

            initialize();
        </script>
    </body>
    </html>
    """
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(file_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save the HTML content to the specified file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    print(f"Interactive HTML report saved to: {file_path}")
    
    
    
def create_violin(df, column_list=None, threshold=0.5, value_column='unsaturation', output_filename='result/report/unsaturation.html'):
    """
    Analyzes a specified value column and creates an interactive HTML violin plot.

    If column_list is provided, it filters rows for each column where the column 
    value is greater than the threshold. 
    
    If column_list is None or False, it uses all the data in the DataFrame 
    to create a single plot.

    Args:
        df (pd.DataFrame): The input DataFrame. Must contain the value_column and a 'name' column.
        column_list (list, optional): A list of column names to process for filtering. 
                                      If None, all data is used. Defaults to None.
        threshold (float): The value threshold for filtering rows. Only used if column_list is provided.
        value_column (str): The name of the column to plot on the y-axis. Defaults to 'unsaturation'.
        output_filename (str): The name of the output HTML file. The directory will be created if it doesn't exist.
    """
    # Ensure the required columns exist before proceeding
    if value_column not in df.columns or 'name' not in df.columns:
        print(f"Error: Required columns '{value_column}' or 'name' not found in DataFrame.")
        return

    plot_df = pd.DataFrame()

    # If a column list is provided, filter the data
    if column_list:
        dfs_to_plot = []
        for col in column_list:
            if col not in df.columns:
                print(f"Warning: Column '{col}' not found in the DataFrame. Skipping.")
                continue

            # Filter the DataFrame and create a copy
            filtered_df = df[df[col] > threshold].copy()

            # Add a 'column' identifier for grouping in the plot
            if not filtered_df.empty:
                filtered_df['column'] = col
                dfs_to_plot.append(filtered_df)
        
        if dfs_to_plot:
            plot_df = pd.concat(dfs_to_plot, ignore_index=True)

    # If no column list is provided, use the entire DataFrame
    else:
        print("No column_list provided. Using all data for the plot.")
        plot_df = df.copy()
        plot_df['column'] = 'All Data'


    # Check if we have any data to plot
    if plot_df.empty:
        print("No data to plot. The resulting DataFrame is empty.")
        return

    # --- Plotting with Plotly ---
    print("Generating interactive plot...")

    # Create the interactive violin plot
    fig = px.violin(
        plot_df,
        x='column',
        y=value_column,
        color='column',  # Color by column for distinction
        color_discrete_sequence=px.colors.qualitative.T10, # Use the tab10 color palette
        box=True,        # Show the box plot inside
        points='all',    # Show all the underlying data points
        hover_name='name', # Use the 'name' column for the main hover label
        hover_data={'column': False, value_column: True}, # Customize hover data
        template='plotly_white' # Use a clean template with a grid
    )

    # Update layout for a cleaner look and to set the y-axis range
    fig.update_layout(
        title_text=None, # Remove the title
        xaxis_title='Groups',
        yaxis_title=f'{value_column.replace("_", " ").title()} Value', # Dynamic Y-axis title
        yaxis_range=[0, plot_df[value_column].max() * 1.1]  # Set y-axis to start at 0
    )

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_filename)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save the plot to an HTML file
    fig.write_html(output_filename)
    print(f"Success! Interactive plot saved to '{output_filename}'")
    
    
    
    


# def create_sunburst(df: pd.DataFrame, filename: str = "result/report/class.html"):
#     """
#     Generates a single, interactive sunburst chart from a DataFrame.

#     Args:
#         df (pd.DataFrame): Must contain 'category' and 'class' columns.
#         filename (str): Output HTML file.
#     """
#     print("Generating single sunburst report for the entire dataset...")

#     if df.empty:
#         print("⚠️ Warning: The input DataFrame is empty. No report will be generated.")
#         return

#     # 1. Create the sunburst chart
#     fig = px.sunburst(
#         df,
#         path=['category', 'class'],
#         color='category',
#         color_discrete_sequence=px.colors.qualitative.Set2,  # 🎨 use a softer color theme
#         height=600,  # slightly smaller
#         width=600    # make square
#     )

#     # 2. Update traces and layout
#     fig.update_traces(
#         hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage of Parent: %{percentParent:.1%}'
#     )
#     fig.update_layout(
#         margin=dict(t=20, l=20, r=20, b=20),
#         font=dict(family="Arial, sans-serif", size=15),
#     )

#     # 3. Save HTML with centering wrapper
#     html_str = fig.to_html(include_plotlyjs='cdn', full_html=False)
#     html_page = f"""
#     <html>
#     <head><meta charset="utf-8"></head>
#     <body style="display:flex; justify-content:center; align-items:center; height:100vh; background-color:#fafafa;">
#         <div style="box-shadow:0 4px 10px rgba(0,0,0,0.1); border-radius:12px; padding:10px; background:white;">
#             {html_str}
#         </div>
#     </body>
#     </html>
#     """

#     os.makedirs(os.path.dirname(filename), exist_ok=True)
#     with open(filename, "w", encoding="utf-8") as f:
#         f.write(html_page)

#     print(f"✅ Report saved successfully to '{filename}'")
# def create_sunburst(df: pd.DataFrame, filename: str = "result/report/class.html"):
#     """
#     Generates a single, interactive sunburst chart from a DataFrame.

#     Args:
#         df (pd.DataFrame): Must contain 'category' and 'class' columns.
#         filename (str): Output HTML file.
#     """
#     print("Generating single sunburst report for the entire dataset...")

#     if df.empty:
#         print("⚠️ Warning: The input DataFrame is empty. No report will be generated.")
#         return

#     # 1. Create the sunburst chart with smaller dimensions
#     fig = px.sunburst(
#         df,
#         path=['category', 'class'],
#         color='category',
#         color_discrete_sequence=px.colors.qualitative.Set2,
#         height=500,  # <-- REDUCED from 600
#         width=1200    # <-- REDUCED from 600
#     )

#     # 2. Update traces and layout for the smaller size
#     fig.update_traces(
#         hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage of Parent: %{percentParent:.1%}'
#     )
#     fig.update_layout(
#         margin=dict(t=30, l=200, r=200, b=30), # <-- REDUCED margins for a tighter fit
#         font=dict(family="Arial, sans-serif", size=14), # <-- SLIGHTLY smaller font size
#     )

#     # 3. Save HTML with centering wrapper
#     html_str = fig.to_html(include_plotlyjs='cdn', full_html=False)
#     html_page = f"""
#     <html>
#     <head><meta charset="utf-8"></head>
#     <body style="display:flex; justify-content:center; align-items:center; height:100vh; background-color:#fafafa;">
#         <div style="box-shadow:0 4px 10px rgba(0,0,0,0.1); border-radius:12px; padding:10px; background:white;">
#             {html_str}
#         </div>
#     </body>
#     </html>
#     """

#     os.makedirs(os.path.dirname(filename), exist_ok=True)
#     with open(filename, "w", encoding="utf-8") as f:
#         f.write(html_page)

#     print(f"✅ Report saved successfully to '{filename}'")
def create_sunburst(df: pd.DataFrame, filename: str = "result/report/class.html"):
    """
    Generates a single, interactive sunburst chart from a DataFrame.

    Args:
        df (pd.DataFrame): Must contain 'category' and 'class' columns.
        filename (str): Output HTML file.
    """
    print("Generating single sunburst report for the entire dataset...")

    if df.empty:
        print("⚠️ Warning: The input DataFrame is empty. No report will be generated.")
        return

    # 1. Create the sunburst chart (remove fixed width/height)
    fig = px.sunburst(
        df,
        path=['category', 'class'],
        color='category',
        color_discrete_sequence=px.colors.qualitative.Set2,
    )

    # 2. Update traces and layout for better fitting
    fig.update_traces(
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage of Parent: %{percentParent:.1%}',
        insidetextorientation="radial"
    )
    fig.update_layout(
        margin=dict(t=40, l=0, r=0, b=0),   # remove extra side margins
        font=dict(family="Arial, sans-serif", size=14),
        autosize=True,
    )

    # 3. Save HTML with a responsive container
    html_str = fig.to_html(include_plotlyjs='cdn', full_html=False)
    html_page = f"""
    <html>
    <head><meta charset="utf-8"></head>
    <body style="display:flex; justify-content:center; align-items:center; height:100vh; background-color:#fafafa; margin:0;">
        <div style="box-shadow:0 4px 10px rgba(0,0,0,0.1); border-radius:12px; padding:10px; background:white; width:100%; height:100%;">
            {html_str}
        </div>
    </body>
    </html>
    """

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_page)

    print(f"✅ Report saved successfully to '{filename}'")





def create_themeriver(
    df: pd.DataFrame, 
    group: list, 
    threshold: float, 
    filename: str = "result/report/themeriver_numeric.html"
):
    """
    Generates an interactive ThemeRiver chart using a numeric (integer) x-axis,
    with x-axis labels annotated by the stage names in 'group'.
    """
    print("--- Generating ThemeRiver Report (Numeric X-Axis) ---")

    if df.empty:
        print("⚠️ Warning: The input DataFrame is empty.")
        return
    if not group:
        print("⚠️ Warning: The 'group' list is empty.")
        return
    
    chart_data = []
    unique_classes = sorted(df['class'].unique())
    
    print(f"\n[DEBUG] Processing {len(group)} stages numerically: {group}")

    for i, stage in enumerate(group, start=1):
        for class_name in unique_classes:
            filtered_df = df[(df['class'] == class_name) & (df[stage] > threshold)]
            count = int(filtered_df.shape[0])
            
            if count > 0:
                chart_data.append([i, count, class_name])

    if not chart_data:
        print("⚠️ Warning: No data points met the threshold criteria. Report will be empty.")

    output_dir = os.path.dirname(filename)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Create JavaScript formatter for axis labels using group names
    group_labels_js = str(group)  # Convert Python list to string representation
    axis_formatter = JsCode(
        f"""function(value) {{
            var labels = {group_labels_js};
            var idx = Math.round(value) - 1;
            if (idx >= 0 && idx < labels.length) {{
                return labels[idx];
            }}
            return '';
        }}"""
    )
        
    c = (
        ThemeRiver(
            init_opts=opts.InitOpts(
                width="1200px", 
                height="600px",
                theme="white",
                bg_color="#ffffff"
            )
        )
        .add(
            series_name=unique_classes,
            data=chart_data,
            singleaxis_opts=opts.SingleAxisOpts(
                type_="value",
                min_=1,
                max_=len(group),
                pos_top="70",
                pos_bottom="50",
                axistick_opts=opts.AxisTickOpts(
                    is_show=True,
                    is_align_with_label=True
                ),
                axislabel_opts=opts.LabelOpts(
                    is_show=True,
                    formatter=axis_formatter,
                    font_size=12,
                    font_weight="bold"
                ),
            ),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title=f"Class Distribution Across Stages (Values > {threshold})",
                pos_left="center"
            ),
            tooltip_opts=opts.TooltipOpts(
                trigger="axis", 
                axis_pointer_type="line"
            ),
            legend_opts=opts.LegendOpts(
                pos_top="5%",
                pos_left="center",
                orient="horizontal"
            )
        )
        .set_colors([
            "#FFF689",
            "#1E2136",
            "#FBC2C2",
            "#32769B",
            "#F4D35E",
            "#64557B",
            "#FFB88A",
            "#62866C",
            "#FF9C5B",
            "#81B2D9",
            "#F67B45",
            "#A0C5E3",
            "#CB7876",
            "#BBA6DD",
            "#8BA47C",
            "#E39B99",
            "#8C7DA8",
            "#B4CFA4"
        ])
    )

    c.render(filename)
    print(f"✅ Report saved successfully to '{filename}'")
    


def create_category_themeriver(
    df: pd.DataFrame, 
    group: list, 
    threshold: float, 
    filename: str = "result/report/category_themeriver_numeric.html"
):
    """
    Generates an interactive ThemeRiver chart for categories using a numeric (integer) x-axis,
    with x-axis labels annotated by the stage names in 'group'.
    Similar to create_themeriver but uses 'category' column instead of 'class'.
    Filters out 'Unknown' category.
    """
    print("--- Generating Category ThemeRiver Report (Numeric X-Axis) ---")

    if df.empty:
        print("⚠️ Warning: The input DataFrame is empty.")
        return
    if not group:
        print("⚠️ Warning: The 'group' list is empty.")
        return
    
    chart_data = []
    # Filter out 'Unknown' category (case-insensitive)
    unique_categories = sorted([cat for cat in df['category'].unique() 
                                if cat and str(cat).lower() != 'unknown'])
    
    print(f"\n[DEBUG] Processing {len(group)} stages numerically: {group}")
    print(f"[DEBUG] Categories (excluding Unknown): {unique_categories}")

    for i, stage in enumerate(group, start=1):
        for category_name in unique_categories:
            filtered_df = df[(df['category'] == category_name) & (df[stage] > threshold)]
            count = int(filtered_df.shape[0])
            
            if count > 0:
                chart_data.append([i, count, category_name])

    if not chart_data:
        print("⚠️ Warning: No data points met the threshold criteria. Report will be empty.")

    output_dir = os.path.dirname(filename)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Create JavaScript formatter for axis labels using group names
    group_labels_js = str(group)  # Convert Python list to string representation
    axis_formatter = JsCode(
        f"""function(value) {{
            var labels = {group_labels_js};
            var idx = Math.round(value) - 1;
            if (idx >= 0 && idx < labels.length) {{
                return labels[idx];
            }}
            return '';
        }}"""
    )
        
    c = (
        ThemeRiver(
            init_opts=opts.InitOpts(
                width="1200px", 
                height="600px",
                theme="white",
                bg_color="#ffffff"
            )
        )
        .add(
            series_name=unique_categories,
            data=chart_data,
            singleaxis_opts=opts.SingleAxisOpts(
                type_="value",
                min_=1,
                max_=len(group),
                pos_top="70",
                pos_bottom="50",
                axistick_opts=opts.AxisTickOpts(
                    is_show=True,
                    is_align_with_label=True
                ),
                axislabel_opts=opts.LabelOpts(
                    is_show=True,
                    formatter=axis_formatter,
                    font_size=12,
                    font_weight="bold"
                ),
            ),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title=f"Category Distribution Across Stages (Values > {threshold})",
                pos_left="center"
            ),
            tooltip_opts=opts.TooltipOpts(
                trigger="axis", 
                axis_pointer_type="line"
            ),
            legend_opts=opts.LegendOpts(
                pos_top="5%",
                pos_left="center",
                orient="horizontal"
            )
        )
        .set_colors([
            "#FFF689",
            "#1E2136",
            "#FBC2C2",
            "#32769B",
            "#F4D35E",
            "#64557B",
            "#FFB88A",
            "#62866C",
            "#FF9C5B",
            "#81B2D9",
            "#F67B45",
            "#A0C5E3",
            "#CB7876",
            "#BBA6DD",
            "#8BA47C",
            "#E39B99",
            "#8C7DA8",
            "#B4CFA4"
        ])
    )

    c.render(filename)
    print(f"✅ Report saved successfully to '{filename}'")





def create_pca(df, groups, output_filename="result/report/pca.html"):
    """
    Performs PCA on a DataFrame and generates an interactive HTML plot, 
    coloring by specified groups and drawing a confidence ellipse for each group.
    Handles missing values by imputing them with the feature mean.

    Args:
        df (pd.DataFrame): The input DataFrame. Rows are features, columns are samples.
                           May contain NaN values.
        groups (list): A list of strings. Each string is a group identifier expected
                       to be found in the column names.
        output_filename (str): The name of the output HTML file.
    """
    # --- 1. Data Preparation ---
    
    # Create a dictionary to hold columns for each group
    group_cols = {group: [] for group in groups}
    
    # Find columns that match each group name
    for group in groups:
        # Filter columns that contain the group name
        matching_cols = [col for col in df.columns if group in col]
        if not matching_cols:
            print(f"Warning: No columns found for group '{group}'. Skipping.")
            continue
        group_cols[group] = matching_cols

    # --- 2. Prepare data for PCA ---
    
    # Combine all relevant columns into a single list for data extraction
    all_feature_columns = [col for sublist in group_cols.values() for col in sublist]
    
    if not all_feature_columns:
        print("Error: No valid columns found for any group. Aborting PCA plot.")
        return

    # Extract the data and create labels (targets) for coloring
    # We transpose the data so that samples are rows and features are columns
    X = df[all_feature_columns].T 
    
    # Create a list of group labels corresponding to each sample (column)
    y = []
    for col_name in X.index:
        for group, cols in group_cols.items():
            if col_name in cols:
                y.append(group)
                break

    # --- 3. Handle Missing Values and Scale Data ---
    
    # Impute missing values using the mean of each feature (column)
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    # Standardize the features before applying PCA
    X_scaled = StandardScaler().fit_transform(X_imputed)

    # --- 4. Perform PCA ---
    
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)

    # Create a DataFrame with the PCA results
    pca_df = pd.DataFrame(
        data=principal_components, 
        columns=['PC1', 'PC2']
    )
    pca_df['Group'] = y
    pca_df['Sample'] = X.index # Add sample names for hover labels

    # --- 5. Plotting with Plotly and Saving to HTML ---
    
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    groups_list = pca_df['Group'].unique()

    for i, group in enumerate(groups_list):
        group_df = pca_df[pca_df['Group'] == group]
        
        # Add scatter trace for the group's data points
        fig.add_trace(go.Scatter(
            x=group_df['PC1'],
            y=group_df['PC2'],
            name=group,
            mode='markers',
            hoverinfo='text',
            hovertext=group_df['Sample'],
            marker=dict(
                color=colors[i % len(colors)],
                size=18,
                line=dict(width=1, color='DarkSlateGrey')
            )
        ))

        # Calculate and draw the confidence ellipse if there are enough points
        if len(group_df) > 2:
            pc1_vals = group_df['PC1']
            pc2_vals = group_df['PC2']
            
            # Calculate ellipse parameters
            cov = np.cov(pc1_vals, pc2_vals)
            mean_x, mean_y = np.mean(pc1_vals), np.mean(pc2_vals)
            
            # Eigenvalues and eigenvectors determine ellipse shape and orientation
            eigvals, eigvecs = np.linalg.eig(cov)
            
            # Sort eigenvalues and eigenvectors
            sort_indices = eigvals.argsort()[::-1]
            eigvals = eigvals[sort_indices]
            eigvecs = eigvecs[:, sort_indices]
            
            # Get angle of rotation and axis lengths
            angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
            # Use 2 standard deviations for a 95% confidence interval
            width, height = 2 * 2 * np.sqrt(eigvals)
            
            # Generate points for the ellipse
            t = np.linspace(0, 2 * np.pi, 100)
            ellipse_x_r = (width / 2) * np.cos(t)
            ellipse_y_r = (height / 2) * np.sin(t)
            
            # Rotate and translate ellipse points
            R = np.array([[np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
                          [np.sin(np.radians(angle)), np.cos(np.radians(angle))]])
            ellipse_rotated = np.dot(R, [ellipse_x_r, ellipse_y_r])
            
            x_ellipse = ellipse_rotated[0] + mean_x
            y_ellipse = ellipse_rotated[1] + mean_y
            
            # Add the ellipse shape as a filled trace
            fig.add_trace(go.Scatter(
                x=x_ellipse,
                y=y_ellipse,
                mode='lines',
                name=f'{group} Ellipse',
                line=dict(color=colors[i % len(colors)], width=1),
                fill='toself',
                opacity=0.2,
                showlegend=False,
                hoverinfo='none'
            ))

    # Customize the plot layout for a cleaner look
    fig.update_layout(
        title='PCA',
        legend_title_text='Groups',
        xaxis_title=f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.2f}%)',
        yaxis_title=f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.2f}%)',
        xaxis=dict(gridcolor='lightgrey'),
        yaxis=dict(gridcolor='lightgrey'),
        plot_bgcolor='white'
    )
    
    # Save the interactive plot to an HTML file
    fig.write_html(output_filename)
    print(f"PCA plot saved to '{output_filename}'")
    
    
    




def create_volcano(
    df,
    p_value_threshold=0.05,
    fc_threshold=1.0,
    result_path="volcano_plot_from_python.html"
):
    """
    Generates a self-contained, interactive HTML file for a volcano plot
    using D3.js with interactive controls.

    The function takes a pandas DataFrame for the initial plot, converts it to JSON,
    and injects it into an HTML template. The generated HTML also contains
    controls to re-filter the plot based on user input.

    Args:
        df (pd.DataFrame): DataFrame for the initial plot. Must include columns:
                           'log2_fold_change', 'p_value', 'class', 'length',
                           'unsaturation'.
        p_value_threshold (float): The initial p-value cutoff for significance.
        fc_threshold (float): The initial absolute log2 fold change cutoff for significance.
        result_path (str): The path where the generated .html file will be saved.
    """
    # --- 1. CONVERT DATAFRAME TO JSON ---
    data_json = df.to_json(orient='records')

    # --- 2. DEFINE THE HTML TEMPLATE ---
    # This template includes controls and the D3.js code.
    # Placeholders will be filled with the initial dataset and thresholds.
    # Note: Curly braces for JS/CSS are escaped by doubling them (e.g., {{ ... }})
    html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive Volcano Plot</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
            display: flex;
            align-items: flex-start;
        }}

        #controls {{
            padding: 20px;
            width: 220px;
            border-right: 1px solid #ddd;
        }}

        #controls h3 {{
            margin-top: 0;
        }}

        #controls div {{
            margin-bottom: 15px;
        }}

        #controls label {{
            display: block;
            margin-bottom: 5px;
            font-size: 14px;
            font-weight: 500;
        }}

        #controls input {{
            width: 100%;
            padding: 8px;
            box-sizing: border-box;
            border: 1px solid #ccc;
            border-radius: 4px;
        }}

        #controls button {{
            width: 100%;
            padding: 10px;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }}

        #controls button:hover {{
            background-color: #0056b3;
        }}

        .tooltip {{
            position: absolute;
            text-align: left;
            padding: 8px 12px;
            font-size: 12px;
            background: #333;
            color: white;
            border: 0px;
            border-radius: 8px;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.2s;
        }}
        
        .axis-label {{ font-size: 14px; font-weight: 500; font-family: sans-serif; }}
        .title-label {{ font-size: 18px; font-weight: 600; font-family: sans-serif; }}
        .legend {{ font-size: 12px; font-family: sans-serif; }}
        .legend-title {{ font-weight: 600; margin-bottom: 5px; }}
        .note-box {{
            font-size: 11px;
            padding: 8px;
            background-color: #f0f8ff; 
            border: 1px solid #e0e0e0;
            border-radius: 5px;
        }}
    </style>
</head>
<body>

<div id="controls">
    <h3>Filters</h3>
    <div>
        <label for="fc-input">Log2 Fold Change</label>
        <input type="number" id="fc-input" value="{fc_threshold}" step="0.1">
    </div>
    <div>
        <label for="pval-input">P-value</label>
        <input type="number" id="pval-input" value="{p_value_threshold}" step="0.01">
    </div>
    <button id="apply-btn">Apply</button>
</div>

<div id="volcano-plot"></div>

<script>
    // --- INITIAL DATA INJECTED FROM PYTHON ---
    const initial_dataset = {data_json};

    // --- Initial plot draw ---
    drawPlot(initial_dataset, {fc_threshold}, {p_value_threshold});

    // --- EVENT LISTENER FOR THE BUTTON ---
    d3.select("#apply-btn").on("click", function() {{
        const fc = parseFloat(d3.select("#fc-input").property("value"));
        const pval = parseFloat(d3.select("#pval-input").property("value"));
        // Re-draw the plot with the original data but new thresholds
        drawPlot(initial_dataset, fc, pval);
    }});

    // --- MAIN PLOTTING FUNCTION ---
    function drawPlot(data, fold_change_threshold, p_value_threshold) {{
        // --- Clear previous plot ---
        d3.select("#volcano-plot").selectAll("*").remove();
        d3.selectAll(".tooltip").remove(); // Also remove old tooltips

        // --- DATA PROCESSING ---
        data.forEach(d => {{
            d['-log10_p_value'] = d.p_value > 0 ? -Math.log10(d.p_value) : 0;
        }});

        // --- CHART SETUP ---
        const margin = {{ top: 60, right: 250, bottom: 60, left: 60 }};
        const width = 800 - margin.left - margin.right;
        const height = 500 - margin.top - margin.bottom;

        const svg = d3.select("#volcano-plot")
            .append("svg")
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom)
            .append("g")
            .attr("transform", `translate(${{margin.left}}, ${{margin.top}})`);
        
        const log_p_threshold = -Math.log10(p_value_threshold);

        // --- SCALES ---
        const x = d3.scaleLinear()
            .domain(d3.extent(data, d => d.log2_fold_change)).nice()
            .range([0, width]);

        const y = d3.scaleLinear()
            .domain([0, d3.max(data, d => d['-log10_p_value'])]).nice()
            .range([height, 0]);

        // --- AXES ---
        svg.append("g").attr("transform", `translate(0, ${{height}})`).call(d3.axisBottom(x));
        svg.append("g").call(d3.axisLeft(y));

        svg.append("text").attr("class", "axis-label").attr("text-anchor", "middle").attr("x", width / 2).attr("y", height + margin.bottom - 15).text("Log2 Fold Change");
        svg.append("text").attr("class", "axis-label").attr("text-anchor", "middle").attr("transform", "rotate(-90)").attr("y", -margin.left + 20).attr("x", -height / 2).text("-log10(p-value)");
        svg.append("text").attr("class", "title-label").attr("text-anchor", "middle").attr("x", width / 2).attr("y", -margin.top / 2 + 10).text("Interactive Volcano Plot");

        // --- THRESHOLD LINES ---
        svg.append("line").attr("x1", x(fold_change_threshold)).attr("x2", x(fold_change_threshold)).attr("y1", 0).attr("y2", height).attr("stroke", "red").attr("stroke-width", 1.5).attr("stroke-dasharray", "4");
        svg.append("line").attr("x1", x(-fold_change_threshold)).attr("x2", x(-fold_change_threshold)).attr("y1", 0).attr("y2", height).attr("stroke", "red").attr("stroke-width", 1.5).attr("stroke-dasharray", "4");
        if (log_p_threshold >= y.domain()[0] && log_p_threshold <= y.domain()[1]) {{
             svg.append("line").attr("x1", 0).attr("x2", width).attr("y1", y(log_p_threshold)).attr("y2", y(log_p_threshold)).attr("stroke", "blue").attr("stroke-width", 1.5).attr("stroke-dasharray", "4");
        }}

        // --- TOOLTIP ---
        const tooltip = d3.select("body").append("div").attr("class", "tooltip");

        // --- DYNAMIC SCALES ---
        const significantData = data.filter(d => Math.abs(d.log2_fold_change) > fold_change_threshold && d.p_value < p_value_threshold);
        const uniqueClasses = [...new Set(data.map(d => d.class))];
        const color = d3.scaleOrdinal(d3.schemeTableau10).domain(uniqueClasses);
        const minDotSize = 5;
        const maxDotSize = 15;
        const sizeScale = d3.scaleSqrt().domain(d3.extent(data, d => d.length)).range([minDotSize, maxDotSize]);
        const alphaScale = d3.scaleLinear().domain(d3.extent(data, d => d.unsaturation)).range([1.0, 0.2]);

        // --- PLOT DATA POINTS ---
        svg.append('g')
            .selectAll("dot")
            .data(data)
            .enter()
            .append("circle")
            .attr("cx", d => x(d.log2_fold_change))
            .attr("cy", d => y(d['-log10_p_value']))
            .attr("r", d => {{
                const isSignificant = Math.abs(d.log2_fold_change) > fold_change_threshold && d.p_value < p_value_threshold;
                return isSignificant ? sizeScale(d.length) || minDotSize : 3;
            }})
            .style("fill", d => {{
                const isSignificant = Math.abs(d.log2_fold_change) > fold_change_threshold && d.p_value < p_value_threshold;
                return isSignificant ? color(d.class) : "grey";
            }})
            .style("stroke", d => {{
                const isSignificant = Math.abs(d.log2_fold_change) > fold_change_threshold && d.p_value < p_value_threshold;
                return isSignificant ? "black" : "none";
            }})
            .style("stroke-width", 0.5)
            .style("opacity", d => {{
                const isSignificant = Math.abs(d.log2_fold_change) > fold_change_threshold && d.p_value < p_value_threshold;
                return isSignificant ? alphaScale(d.unsaturation) : 0.5;
            }})
            .on("mouseover", function(event, d) {{
                tooltip.transition().duration(200).style("opacity", .9);
                tooltip.html(
                    `<strong>Name:</strong> ${{d.name}}<br/>` +
                    `<strong>Log2 FC:</strong> ${{d.log2_fold_change.toFixed(2)}}<br/>` +
                    `<strong>P-value:</strong> ${{d.p_value.toExponential(2)}}`
                )
                .style("left", (event.pageX + 15) + "px")
                .style("top", (event.pageY - 28) + "px");
                d3.select(this).style("stroke-width", 1.5).style("stroke", "black");
            }})
            .on("mouseout", function(event, d) {{
                tooltip.style("opacity", 0);
                const isSignificant = Math.abs(d.log2_fold_change) > fold_change_threshold && d.p_value < p_value_threshold;
                d3.select(this).style("stroke-width", isSignificant ? 0.5 : 0);
            }});

        // --- LEGENDS ---
        const legendX = width + 40;
        const significantClassCounts = d3.rollup(significantData, v => v.length, d => d.class);
        const legendClasses = uniqueClasses.filter(cls => (significantClassCounts.get(cls) || 0) >= 1);
        const classLegend = svg.append("g").attr("class", "legend").attr("transform", `translate(${{legendX}}, 0)`);
        classLegend.append("text").attr("class", "legend-title").text("Metabolite Class");
        
        const numClasses = legendClasses.length;
        const isMultiColumn = numClasses > 10;
        const midPoint = Math.ceil(numClasses / 2);
        const columnWidth = 120; 

        legendClasses.forEach((cls, i) => {{
            const xPos = isMultiColumn && i >= midPoint ? columnWidth : 0;
            const yIndex = isMultiColumn ? (i % midPoint) : i;
            const yPos = 20 * (yIndex + 1);
            const legendRow = classLegend.append("g").attr("transform", `translate(${{xPos}}, ${{yPos}})`);
            legendRow.append("rect").attr("width", 15).attr("height", 15).attr("fill", color(cls));
            legendRow.append("text").attr("x", 20).attr("y", 12).text(cls);
        }});
        
        const legendHeight = isMultiColumn ? midPoint : numClasses;
        const sizeLegendY = 20 * (legendHeight + 2);
        const sizeLegend = svg.append("g").attr("class", "legend").attr("transform", `translate(${{legendX}}, ${{sizeLegendY}})`);
        sizeLegend.append("text").attr("class", "legend-title").text("Length (Size)");
        
        const sizeValues = d3.extent(data, d => d.length);
        let sizeLegendData = [];
        if (sizeValues[0] !== undefined && sizeValues[1] !== undefined) {{
            const midVal = (sizeValues[0] + sizeValues[1]) / 2;
            const manualTicks = [sizeValues[0], midVal, sizeValues[1]];
            sizeLegendData = manualTicks.map(val => ({{
                label: Math.round(val),
                size: sizeScale(val)
            }}));
        }}

        sizeLegendData.forEach((item, i) => {{
            const legendRow = sizeLegend.append("g").attr("transform", `translate(10, ${{25 * (i + 1)}})`);
            legendRow.append("circle").attr("r", item.size).attr("fill", "gray");
            legendRow.append("text").attr("x", 20).attr("y", 4).text(item.label);
        }});

        const noteY = sizeLegendY + 25 * (sizeLegendData.length + 1) + 20;
        svg.append("foreignObject")
            .attr("width", 200).attr("height", 100)
            .attr("transform", `translate(${{legendX - 10}}, ${{noteY}})`).append("xhtml:div")
            .attr("class", "note-box")
            .html("A point's transparency shows its unsaturation level.<br><b>More transparent = Higher unsaturation</b>");
    }}
</script>

</body>
</html>
    """

    # --- 3. WRITE THE POPULATED HTML TO A FILE ---
    try:
        with open(result_path, 'w', encoding='utf-8') as f:
            f.write(html_template)
        print(f"Interactive volcano plot saved to '{result_path}'")
    except IOError as e:
        print(f"Error writing to file: {e}")
        
        
