Task 3: Cloud-Native Data Processing with Serverless Functions (Simulated)
Overview

In this task, I implemented a simulated serverless data processing workflow using:

Azurite (local Azure Blob Storage emulator)

Docker to run Azurite

Python to simulate a serverless function

JSON file to simulate a NoSQL database

The objective was to process the All_Diets.csv dataset stored in Azurite and calculate average macronutrient values per diet type.

Files Included (Task3_Final.tar.gz):
upload_to_azurite.py

Uploads All_Diets.csv to Azurite Blob Storage.
process_from_azurite.py

Simulates serverless processing logic.
requirements.txt

Required Python dependencies:
azure-storage-blob
pandas
All_Diets.csv

Input dataset.
results.json
Output file (simulated NoSQL database).