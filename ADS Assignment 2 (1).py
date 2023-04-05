"""
This code plots different visualisations methods that is bar plot plot, line plot 
and heat map using single dataset based on multiple index.

"""

# importing required packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import skew
from scipy.stats import kurtosis



def data(x):
    """
    This function takes a pandas DataFrame `a` and performs several statistical calculations on the columns.
    It prints the summary statistics, correlation matrix, skewness, and kurtosis
    for the selected columns.

    Parameters
    ----------
    x : String
        a string that represents the name of the CSV file to be read

    Returns
    -------
    worldbank (pd.DataFrame): the pandas DataFrame that contains the data from the CSV file
    transpose (pd.DataFrame): the transposed pandas DataFrame
    
    """
    

    # create a file path using the input string
    address = r'C:\Users\Diraj\Downloads' + x

    # read the csv file into a pandas DataFrame using the file path
    worldbank = pd.read_csv(address)
    # transpose the DataFrame
    transpose = worldbank.transpose

    return(worldbank, transpose)


data('\worldbank.csv')

data = pd.read_csv(r'C:\Users\Diraj\Downloads\worldbank.csv')


def clean(x):
    """
    Cleans the input DataFrame by check the sum of null values and filling the null values with 0.

    Parameters
    ----------
    x : pandas DataFrame
        the DataFrame to be cleaned.

    Returns
    -------
    Cleaned data

    """

    # count the number of missing values in each column of the DataFrame
    x.isnull().sum()

    # filling the missing values by zero
    x.fillna(0, inplace=True)

    return


clean(data)


def stats(a):
    """
    This function takes a pandas DataFrame `a` and performs several statistical calculations on the columns.
    It prints the summary statistics, correlation matrix, skewness, and kurtosis
    for the selected columns.

    Parameters
    ----------
    a : pandas DataFrame
        input the datafarame to perform different statistical functions
        

    Returns
    -------
    None.

    """

    # extract the columns from the 5th column onward and assign to variable "stats"
    stats = a.iloc[:, 4:]

    # calculate the skewness,kurtosis and Covariance
    print('Skewness :','\n',skew(stats, axis=0, bias=True))
    print('Kurtosis :','\n',kurtosis(stats, axis=0, bias=True))
    print('Describe :', '\n',stats.describe())
    print('Covariance :','\n',stats.cov())

#Calling function to calculate statistical functions
stats(data)


def nitrous_bar(b):
    """
    This function takes a pandas DataFrame containing data from worldbank 
    climate change dataset and creates a barplot of the % change in Nitrous 
    oxide emissions from 1990
    
    Parameters
    ----------
    b : pandas DataFrame
        Passes the values of the selected countries Nitrous emissions.

    Returns
    -------
    This function plots the barplot for % of Nitrous oxide emissions

    """
    
    

    # Select rows where the "Indicator Name" column is "Nitrous oxide emissions (% change from 1990)"
    Nitrous = b[b['Indicator Name'] ==
                'Nitrous oxide emissions (% change from 1990)']

    # Select rows where the "Country Name" column is one of a list of countries
    Nitrous_emission = Nitrous[Nitrous['Country Name'].isin(['Australia', 'Brazil', 'China', 'India', 'New Zealand',
                                                             'Brazil', 'Korea, Rep.', 'Sweden'])]

    # Define the width of each bar
    bar_width = 0.1

    # Define the positions of the bars on the x-axis
    r1 = np.arange(len(Nitrous_emission))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]
    r5 = [x + bar_width for x in r4]
    r6 = [x + bar_width for x in r5]
    r7 = [x + bar_width for x in r6]

    # Create a bar plot of the selected data, with a different color for each year
    plt.subplots(figsize=(15, 8))
    plt.bar(r1, Nitrous_emission['1991'], color='aquamarine',
            width=bar_width, edgecolor='black', label='1991')
    plt.bar(r2, Nitrous_emission['1994'], color='turquoise',
            width=bar_width, edgecolor='black', label='1994')
    plt.bar(r3, Nitrous_emission['1997'], color='lightseagreen',
            width=bar_width, edgecolor='black', label='1997')
    plt.bar(r4, Nitrous_emission['2000'], color='mediumturquoise',
            width=bar_width, edgecolor='black', label='2000')
    plt.bar(r5, Nitrous_emission['2003'], color='lightcyan',
            width=bar_width, edgecolor='black', label='2003')
    plt.bar(r6, Nitrous_emission['2007'], color='paleturquoise',
            width=bar_width, edgecolor='black', label='2007')
    plt.bar(r7, Nitrous_emission['2011'], color='mediumaquamarine',
            width=bar_width, edgecolor='black', label='2011')

    # Set the x-tick labels to the country names
    plt.xticks([r + bar_width*2 for r in range(len(Nitrous_emission))],
               Nitrous_emission['Country Name'])

    # Adding labels to the axis
    plt.xlabel('Countries', fontweight='bold', fontsize=15)
    plt.ylabel('Nitrous_emission', fontweight='bold', fontsize=15)
    plt.title('Nitrous oxide emissions (% change from 1990)', fontweight='bold', fontsize=15)
    plt.legend()
    plt.savefig('bar_plot_nitrous.png')
    plt.show()


def Electricity_prod(c):
    """
    This function takes a pandas DataFrame `c` containing data on worldbank 
    climate change data and creates a bar plot of the percentage change in 
    Electricity production from coal sources (% of total) for a selection of 
    countries.

    Parameters
    ----------
    c : pandas DataFrame
        Passes the values of the selected countries Nitrous emissions.

    Returns
    -------
    This function plots the barplot for Electricity production from coal sources

    """

    # Select rows where the "Indicator Name" column is "Electricity production from coal sources (% of total)"
    electricity = c[c['Indicator Name'] ==
                    'Electricity production from coal sources (% of total)']

    # Select rows where the "Country Name" column is one of a list of countries
    electricity_prod = electricity[electricity['Country Name'].isin(['Australia', 'Brazil', 'China', 'India', 'New Zealand',
                                                                     'Brazil', 'Korea, Rep.', 'Sweden'])]

    # Define the width of each bar
    bar_width = 0.1

    # Define the positions of the bars on the x-axis
    r1 = np.arange(len(electricity_prod))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]
    r5 = [x + bar_width for x in r4]
    r6 = [x + bar_width for x in r5]
    r7 = [x + bar_width for x in r6]

    # Create a bar plot of the selected data, with a different color for each year
    plt.subplots(figsize=(15, 8))
    plt.bar(r1, electricity_prod['1993'], color='mistyrose',
            width=bar_width, edgecolor='black', label='1993')
    plt.bar(r2, electricity_prod['1996'], color='salmon',
            width=bar_width, edgecolor='black', label='1996')
    plt.bar(r3, electricity_prod['1999'], color='tomato',
            width=bar_width, edgecolor='black', label='1999')
    plt.bar(r4, electricity_prod['2002'], color='darksalmon',
            width=bar_width, edgecolor='black', label='2002')
    plt.bar(r5, electricity_prod['2005'], color='coral',
            width=bar_width, edgecolor='black', label='2005')
    plt.bar(r6, electricity_prod['2008'], color='lightsalmon',
            width=bar_width, edgecolor='black', label='2008')
    plt.bar(r7, electricity_prod['2011'], color='sienna',
            width=bar_width, edgecolor='black', label='2011')
    # Set the x-tick labels to the country names
    plt.xticks([r + bar_width*2 for r in range(len(electricity_prod))],
               electricity_prod['Country Name'])
    # Adding labels to the axis
    plt.xlabel('Countries', fontweight='bold', fontsize=15)
    plt.ylabel('electricity production from coal sources', fontweight='bold', fontsize=15)
    plt.title('Electricity production from coal sources (% of total)',
              fontweight='bold', fontsize=15)
    plt.legend()
    plt.savefig('bar_plot.png')
    plt.show()

# Calls the functions to create the barplot
nitrous_bar(data)
Electricity_prod(data)


def line_plot(c):
    """
    Plots a line graph of forest area (%) of selected countries from the given 
    dataframe

    Parameters
    ----------
    c : Pandas dataframe
        Pandas dataframe containing the worldbank data of Methane emissions.

    Returns
    -------
    This function plots the lineplot for Methane emissions

    """
    
    # filtering out the data related to forest area for selected countries
    methane_emissions = c[c['Indicator Name'] ==
                          'Methane emissions (% change from 1990)']

    methane = methane_emissions[methane_emissions['Country Name'].isin(['Australia', 'Brazil', 'China', 'India', 'New Zealand',
                                                                        'Brazil', 'Korea, Rep.', 'Sweden'])]

    # creating transpose of the filtered data
    Trans = methane.transpose()
    Trans.rename(columns=Trans.iloc[0], inplace=True)
    methane_transpose = Trans.iloc[4:]

    # Replacing the null values by zeros
    methane_transpose.fillna(0, inplace=True)

    # plotting the line graph
    plt.figure(figsize=(22, 10))
    plt.plot(methane_transpose.index,
             methane_transpose['Australia'], linestyle='dashed', label='Australia')
    plt.plot(methane_transpose.index,
             methane_transpose['Sweden'], linestyle='dashed', label='Sweden')
    plt.plot(methane_transpose.index,
             methane_transpose['China'], linestyle='dashed', label='China')
    plt.plot(methane_transpose.index,
             methane_transpose['India'], linestyle='dashed', label='India')
    plt.plot(methane_transpose.index,
             methane_transpose['New Zealand'], linestyle='dashed', label='New Zealand')
    plt.plot(methane_transpose.index,
             methane_transpose['Brazil'], linestyle='dashed', label='Brazil')
    plt.plot(methane_transpose.index,
             methane_transpose['Korea, Rep.'], linestyle='dashed', label='Korea, Rep.')

    # Setting x limit
    plt.xlim('1995', '2012')
    # Adding labels to the axis
    plt.xlabel('Year', fontsize=20, fontweight='bold')
    plt.ylabel('Methane emissions', fontsize=20, fontweight='bold')
    plt.title('Methane emissions (% change from 1990)',
              fontsize=20, fontweight='bold')
    plt.savefig('line_plot_methane.png')
    plt.legend()
    plt.show()


def Pol_line_plot(c):
    """
    This function takes a pandas dataframe 'c' as input and plots a line graph showing the urban population growth
    annual percentage for 10 different countries.

    Parameters
    ----------
    c : Pandas DataFrame
        DESCRIPTION.

    Returns
    -------
    This function plots the lineplot for Population growth (annual %)

    """

    # filter the rows which have indicator name as 'Urban population growth (annual %)'
    pol = c[c['Indicator Name'] == 'Population growth (annual %)']

    # filter the rows for the 10 selected countries
    population = pol[pol['Country Name'].isin(['Australia', 'Brazil', 'China', 'India', 'New Zealand',
                                                                'Brazil', 'Korea, Rep.', 'Sweden'])]

    # # creating transpose of the filtered data
    Tran = population.transpose()
    Tran.rename(columns=Tran.iloc[0], inplace=True)
    population_transpose = Tran.iloc[4:]
    # Replacing the null values by zeros
    population_transpose.fillna(0, inplace=True)

    # plotting the line graph
    plt.figure(figsize=(22, 10))
    plt.plot(population_transpose.index,
             population_transpose['Australia'], linestyle='dashed', label='Australia')
    plt.plot(population_transpose.index,
             population_transpose['Sweden'], linestyle='dashed', label='Sweden')
    plt.plot(population_transpose.index,
             population_transpose['China'], linestyle='dashed', label='China')
    plt.plot(population_transpose.index,
             population_transpose['India'], linestyle='dashed', label='India')
    plt.plot(population_transpose.index,
             population_transpose['New Zealand'], linestyle='dashed', label='New Zealand')
    plt.plot(population_transpose.index,
             population_transpose['Brazil'], linestyle='dashed', label='Brazil')
    plt.plot(population_transpose.index,
             population_transpose['Korea, Rep.'], linestyle='dashed', label='Korea, Rep.')
    # setting x limit
    plt.xlim('2001', '2012')
    # adding labels to the axis
    plt.xlabel('Year', fontsize=20, fontweight='bold')
    plt.ylabel('percentage of population growth', fontsize=20, fontweight='bold')
    plt.title('Population growth (annual %)',
              fontsize=20, fontweight='bold')
    plt.legend(loc='best')
    plt.savefig('line_plot_population.png')
    plt.show()

# Calls the functions to create the lineplot
line_plot(data)
Pol_line_plot(data)


def heatmap_brazil(x):
    """
    A function that creates a heatmap of the correlation matrix between 
    different indicators for Brazil.

    Parameters
    ----------
    x : Pandas DataFrame
        A DataFrame containing data on different indicators for various countries.

    Returns
    -------
    This function plots the heatmap of Brazil

    """

    # Specify the indicators to be used in the heatmap
    indicator = ['Nitrous oxide emissions (% change from 1990)',
                 'Electricity production from coal sources (% of total)',
                 'Population growth (annual %)',
                 'Agricultural land (% of land area)',
                 'Forest area (% of land area)',
                 'Methane emissions (% change from 1990)']

    # Filter the data to keep only China's data and the specified indicators
    braz = x.loc[x['Country Name'] == 'Brazil']
    brazil = braz[braz['Indicator Name'].isin(indicator)]

    # Pivot the data to create a DataFrame with each indicator as a column
    brazil_df = brazil.pivot_table(brazil, columns=x['Indicator Name'])

    # Compute the correlation matrix for the DataFrame
    brazil_df.corr()

    # Plot the heatmap using seaborn
    plt.figure(figsize=(12, 8))
    sns.heatmap(brazil_df.corr(), fmt='.2g', annot=True,
                cmap='magma', linecolor='black')
    plt.title('Brazil', fontsize=25, fontweight='bold')
    plt.xticks(fontsize=10, fontweight='bold')
    plt.yticks(fontsize=10, fontweight='bold')
    plt.xlabel('')
    plt.ylabel('')
    plt.savefig('heatmap_brazil.png')
    plt.show()


def heatmap_china(x):
    """
    A function that creates a heatmap of the correlation matrix between 
    different indicators for China.

    Parameters
    ----------
    x : Pandas DataFrame
        A DataFrame containing data on different indicators for various countries.

    Returns
    -------
   This function plots the heatmap for China.

    """
    

    # Specify the indicators to be used in the heatmap
    indicator = ['Nitrous oxide emissions (% change from 1990)',
                 'Electricity production from coal sources (% of total)',
                 'Population growth (annual %)',
                 'Agricultural land (% of land area)',
                 'Forest area (% of land area)',
                 'Methane emissions (% change from 1990)']

    # Filter the data to keep only Afghanistan's data and the specified indicators
    chi = x.loc[x['Country Name'] == 'China']
    China = chi[chi['Indicator Name'].isin(indicator)]

    # Pivot the data to create a DataFrame with each indicator as a column
    China_df = China.pivot_table(
        China, columns=x['Indicator Name'])
    # Compute the correlation matrix for the DataFrame
    China_df.corr()
    # Plot the heatmap using seaborn
    plt.figure(figsize=(12, 8))
    sns.heatmap(China_df.corr(), fmt='.2g',
                annot=True, cmap='mako', linecolor='black')
    plt.title('China', fontsize=25, fontweight='bold')
    plt.xticks(fontsize=10, fontweight='bold')
    plt.yticks(fontsize=10, fontweight='bold')
    plt.xlabel('')
    plt.ylabel('')
    plt.savefig('heatmap_china.png')
    plt.show()

# Calls the functions to create the heatmap
heatmap_brazil(data)
heatmap_china(data)
