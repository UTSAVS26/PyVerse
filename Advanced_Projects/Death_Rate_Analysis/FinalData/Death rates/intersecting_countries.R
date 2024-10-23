#Extractign 2004 gender +country data
library(httr)
library(jsonlite)
#Using API to extract data from WHO website
res <- GET("https://ghoapi.azureedge.net/api/Indicator?$filter=contains(IndicatorName,%20%27Death%27)")
data <- res$content
data <- rawToChar(data)
data <- fromJSON(data)
#Extracting the interesting death causes from data
i<-array(c(49,50,51,52 ,53,54,55,57,58,59,60,61,62,63,64,65,66 ,74))
mat <- list(0)

for(i in 1:18){
  mat[[i]] <- unique(Final_data_corrected[[i]][,1])
}
mat
#Subseting all the dataframes by which all the dataframes contain commom countries.
a <- Reduce(intersect, mat)

# df= Final_data_corrected[[1]]
f1 = function(df){
  index = which(df$SpatialDim %in% a == T) 
  df1 = df[index,]
  return(df1)
  
}

Final_data_corrected_1 = lapply(Final_data_corrected,f1)

save(Final_data_corrected_1, file = 'Final_data_2.Rdata')



