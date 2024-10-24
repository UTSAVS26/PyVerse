#Installing packages
#loading libraries
#loading data
install.packages(c("tidyverse","rvest","shiny","jsonlite","httr","ggplot2","dplyr","plotly","ggpubr","gridExtra","patchwork"))
library(tidyverse)
library(jsonlite)
library(rvest)
library(shiny)
library(httr)
library(ggplot2)
library(dplyr)
library(plotly)
library(ggpubr)
library(gridExtra)
library(patchwork)
load("FinalData/Death rates/Final_data_3.Rdata") #Final_data_corrected_2
load("FinalData/HDI/developing.Rdata") #developing
load("FinalData/HDI/developed.Rdata") #developed
HDI <- get(load("FinalData/HDI/HDI.Rdata")) 
developed_gdp <- get(load("FinalData/GDP/Developed_countries_gdp.Rdata"))
developing_gdp <- get(load("FinalData/GDP/Developing_countries_gdp.Rdata"))
load("FinalData/Reigons/ListOfCountriesContinentWise.Rdata") #listcc
load("Raw Data/data.Rdata")
