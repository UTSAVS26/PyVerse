#Extractign 2004 gender +country data
library(httr)
library(jsonlite)
#Using API to extract data from WHO website
res <- GET("https://ghoapi.azureedge.net/api/Indicator?$filter=contains(IndicatorName,%20%27Death%27)")
data <- res$content
data <- rawToChar(data)
data <- fromJSON(data)
i<-array(c(53,54,55,56,57,58,59,61,62,63,64,65,66,67,68,69,70))
extractData <- function(n)
{
  link <- paste0("https://ghoapi.azureedge.net/api/",data[["value"]][["IndicatorCode"]][n],"?$filter=TimeDim%20eq%202004")
  indiRes <- GET(link)
  indiData <- fromJSON(rawToChar(indiRes$content))
  return(indiData)
}
#Merge all the dataframes or different causesa into a single list. 
Final_data = list(NULL)

for( x in 1 : length(i) )
{
  df<-extractData(i[x])
  df <- df[["value"]]
  COLs <-array(c(4,8,16))
  df<-df[,COLs]
  Final_data[[x]] = df
  # names(Final_data[[x]]) = paste0(data[["value"]][["IndicatorCode"]][x],"_2004")
  # save(df,file=paste0(data[["value"]][["IndicatorCode"]][x],"_2004.Rdata"))
}
save(Final_data, file = 'Final_data_0.Rdata')
