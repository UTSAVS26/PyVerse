#Extractign 2004 gender +country data
library(httr)
library(jsonlite)
#Using API to extract data from WHO website
res <- GET("https://ghoapi.azureedge.net/api/Indicator?$filter=contains(IndicatorName,%20%27Death%27)")
data <- res$content
data <- rawToChar(data)
data <- fromJSON(data)
i<-array(c(49,50,51,52 ,53,54,55,57,58,59,60,61,62,63,64,65,66 ,74))
mat <- list(0)

for(i in 1:18){
  mat[[i]] <- unique(Final_data_corrected[[i]][,1])
}

a <- Reduce(intersect, mat)
#Extracting the country codes
viewCC <- function()
{
  ccres <- GET("https://ghoapi.azureedge.net/api/DIMENSION/COUNTRY/DimensionValues")
  ccdata <- ccres$content
  ccdata <- rawToChar(ccdata)
  ccdata <- fromJSON(ccdata)
  df <- data.frame(ccdata$value)
  df <- data.frame(df$Code,df$Title,df$ParentTitle)
  return(df)
}
viewCC()[,1]
View(viewCC())
# df= Final_data_corrected[[1]]
f1 = function(df){
  index = which(df$SpatialDim %in% a == T) 
  df1 = df[index,]
  return(df1)
  
}

# Adding the Continent and countries column to all the dataframes.
Final_data_corrected_1 = lapply(Final_data_corrected,f1)
f2 = function(df){
  index = viewCC()[,1][which(viewCC()[,1] %in% df$SpatialDim == T)]
  #df2 = viewCC()[viewCC()[,1] == index,]
  country = viewCC()[,2][which(viewCC()[,1] %in% df$SpatialDim == T)]
  continent = viewCC()[,3][which(viewCC()[,1] %in% df$SpatialDim == T)]
  colnames(df) = c("index","gender","Death_Rate")
  d = data.frame(index,country,continent)
  df2 = merge(df,d,by = "index")
  #country = viewCC()[,2][index]
  #continent = viewCC()[,3][index]
  #df2 = data.frame(df,df2)
  return(df2)
}
Final_data_corrected_2 = lapply(Final_data_corrected_1,f2)
View(f2(Final_data_corrected_1[[1]]))

View(Final_data_corrected_2)
save(Final_data_corrected_1, file = 'Final_data_2.Rdata')
save(Final_data_corrected_2, file = 'Final_data_3.Rdata')
