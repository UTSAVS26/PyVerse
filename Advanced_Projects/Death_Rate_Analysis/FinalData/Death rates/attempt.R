#Extractign 2004 gender +country data
library(httr)
library(jsonlite)
#Using API to extract data from WHO website
res <- GET("https://ghoapi.azureedge.net/api/Indicator?$filter=contains(IndicatorName,%20%27Death%27)")
data <- res$content
data <- rawToChar(data)
data <- fromJSON(data)
i<-array(c(54,55,56,57,58,59,61,62,63,64,65,66,67,68,69,70))

# df = Final_data[[1]]

f = function(df){
  a = names(which(table(df$SpatialDim) < 3))
  
  index = which(df$SpatialDim %in% a == T) 
  
  # df1 = ifelse(length(index) == 0, df, df[-index,])
  if(length(index) == 0){
    df1 = df
  }else
    df1 = df[-index,]
  
  return(df1)
}

f(Final_data[[2]])
View(f(Final_data[[2]]))
Final_data_corrected = lapply(Final_data,f)

save(Final_data_corrected, file = 'Final_data_1.Rdata')

