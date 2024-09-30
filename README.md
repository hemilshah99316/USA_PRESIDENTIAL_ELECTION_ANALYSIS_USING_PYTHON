# US Presidential Election Analysis (1976-2020)🎉

## Overview 📊

This project provides a comprehensive exploratory data analysis (EDA) of the United States presidential elections from 1976 to 2020. Utilizing data gathered from each election, we aim to uncover significant trends, patterns, and insights that inform political strategies and enhance public understanding of the electoral process.

## Objectives 🎯

1. **Election Results Analysis**: Examine the results across different political parties over the years. 🗳️
2. **State-Level Voting Patterns**: Investigate voting trends to identify key swing states and their impact on election outcomes. 🔍
3. **Influential Candidates**: Highlight the most notable individuals in presidential candidacy history. 🏛️

## Data Sources 📁

This analysis leverages comprehensive election data, enriched by incorporating demographic, economic, and polling data to better understand voter behavior.

## Data Structure 🗃️

| Column Name        | Data Type | Description                                                    |
|--------------------|-----------|----------------------------------------------------------------|
| year               | Integer   | The year of the election.                                     |
| state              | String    | The name of the state where the election took place.         |
| state_po           | String    | The postal abbreviation for the state.                       |
| state_fips         | Integer   | The Federal Information Processing Standards code for the state. |
| state_cen          | Integer   | The Census Bureau code for the state.                        |
| state_ic           | Integer   | The state identifier code used for electoral purposes.       |
| office             | String    | The office contested in the election (e.g., US President).  |
| candidate          | String    | The name of the candidate running for office.                |
| party_detailed     | String    | The full name of the party affiliated with the candidate.    |
| writein            | Boolean   | Indicates whether the candidate was a write-in candidate.    |
| candidatevotes     | Integer   | The number of votes received by the candidate.               |
| totalvotes         | Integer   | The total number of votes cast in the election.              |
| version            | String    | The version of the dataset.                                   |
| notes              | String    | Additional notes regarding the election data.                |
| party_simplified   | String    | A simplified category of the party (e.g., Democrat, Republican). |

## Visualization Techniques 📈

We utilized various libraries to present our findings through informative visualizations:
- **Bokeh**: For interactive visualizations. 📊
- **Matplotlib & Seaborn**: For traditional plotting and statistical visualizations. 🎨
- **Holoviews**: For advanced data representation and interactivity. 🌟

## Tools and Libraries Used 🛠️

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bokeh.plotting import figure, show
import holoviews as hv
from holoviews import opts
```

## Key Findings 🔑

- **Major States**: New York and California have shown the highest total contributions in elections over the past 44 years. 📈
- **Influential Leaders**: Donald J. Trump and Barack H. Obama emerged as the most impactful figures in recent electoral history. 👔
- **Party Dominance**: The Democratic and Republican parties continue to dominate the electoral landscape, with Democrats holding a slight edge. 🎉
- **Diversity of Parties**: Colorado recorded the most unique parties participating in elections, followed closely by New York. 🌈
- **Swing States**: Key battleground states such as Pennsylvania, Wisconsin, and Illinois are crucial in determining electoral outcomes. ⚖️

## Future Work 🚀

- **Predictive Modeling**: Developing methodologies to forecast future election outcomes based on historical data. 🔮
- **Enhanced Visualization**: Exploring advanced visualization technique, including interactive dashboards and virtual reality environments.🌐
- **Bias Mitigation**: Investigating potential discrepancies in data collection and proposing methods to improve accuracy. 🕵️‍♂️

## Conclusion 🏁

This analysis not only enhances our understanding of historical electoral dynamics but also serves as a resource for strategists, policymakers, and the public. By shedding light on past trends, we aim to guide future policy decisions and foster greater democratic participation. 🗽

## Getting Started 🏗️

To run the analysis locally, ensure you have the necessary libraries installed. You can clone this repository and execute the Jupyter Notebook provided.

```bash
git clone https://github.com/hemilshah99316/USA_PRESIDENTIAL_ELECTION_ANALYSIS_USING_PYTHON.git
```
OR
Download Zip File

## Requirements 📋

To run this project, ensure you have the following packages installed:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `bokeh`
- `holoviews`
- `scipy`

You can install the required libraries using pip:

```bash
pip install pandas numpy matplotlib seaborn bokeh holoviews scipy
```

## How to Run 🚀

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/hemilshah99316/USA_PRESIDENTIAL_ELECTION_ANALYSIS_USING_PYTHON.git
   ```

2. Open the Jupyter Notebook:
   ```bash
   jupyter notebook us-presidential-eda-from-1976.ipynb
   ```

3. Alternatively, you can run the Python script directly:
   ```bash
   python us-presidential-eda-from-1976.py
   ```

4. View the presentations and reports in PDF format for detailed insights:
   - `Project_Presentation USA Presidential EDA.pptx`
   - `Report.pdf`
   - `USA Presidential Election EDA PDF.pdf`

