
# Foreign Exchange Rate Prediction Algorithm Comparison  

Akin Wilson, University of Nottingham 

coursework-akinolawilson created by GitHub Classroom

### Overview 
Statistical machine learning has been applied to financial markets since the dawn of the computer age. The foreign exchange market in particular due to its accessibility, liquidity and trading volume is a perfect contender to apply quantitative trading. Bar weekends, trading in this market is conducted on a continuous basis all around the globe, leading to stable numerical data and low transaction fees in comparison to other markets such as the equity, fixed-income and derivatives market. 

The currency pair that will be analysed is the USD/GBP. The current geopolitical uncertainties surrounding the pair pose an interesting environment to attempt to predict their exchange rate. Both parametric and non-parametric models will be employed and compared in the investigation. 

### Sourcing High-Quality Data
Quality data is key to ensure predictive statistical methods provide accurate results. The data for this investigation will be gathered directly from a Bloomberg Terminal. It is also important to ensure that the relative accompanying data used to aid the algorithms performance are relevant. Selecting these additional features will be based on feature selection techniques such as univariate selection, feature importance and a Correlation Matrix accompanied by a Heatmap. 

### Libraries
The libraries that will be employed throughout the investigation are:
* Pandas for data handling 
* Scikit-learn for feature selecting and statistical algorithms
* Matplotlib for visualisation
* Numpy for numerical operations
* Plotly for visualisation


### Purpose of Investigation 
Once shown that using a variety of algorithms can lead to favourable results, the program resulting from the investigation will be developed upon with the aim of building an autonomous trading agent. An agent based on reinforcement learning for foreign exchange could potential trading through volatile periods with success given its adaptability. 

A final comparison between the effectiveness of each algorithm will be made and commented on. Any additional potential modification to the algorithms with the aim of improving efficiency will also be explored, once their vanilla counterparts have been applied and tested.

### File Information and Content
The file name dataCleaningAndPreparing.py contains all the data cleansing and feature evaluation of the project. The file trainingAndTest.py contains the fitting and latter testing of four different statistical methods. The PDF file is a report of the project highlighting any project specific coding choices made in this investigation. Lastly, the zipped file FXdataSet contains all the data sets needed to run the previously mentioned files. 
