# data-trimmer

Trims the constant valued columns.  Also removes rows from the start and the end of a dataset with features with consecutive zeroes or a fixed number of rows. Save a CSV with removed files and columns for applying the same trimming to another dataset.
6.2.1	INTEGRATION REQUIREMENTS
The integration tests for each interaction requirement can be found on Annex 1. This component´s requirements are the following: 
•	Use data generated from MQL4 Data Exporter (same as 6.1).
•	Use data generated from data-window.
6.2.2	COMPONENT REQUIREMENTS
The component tests for each interaction requirement can be found on Annex 1. The requirements of this component are the following: 
•	Configurable window size via input parameters. 
•	Must trim a configurable number of rows from the start or end of the output dataset.
•	Optionally auto trim the dataset: Trims the constant valued columns.  Also removes rows from the start and the end of a dataset with features with consecutive zeroes.
•	Save csv with a list of the columns and rows trimmed. 
•	Trim from a saved csv list of files and columns to trim.

Pending  proper documentation.
