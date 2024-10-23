library(httr)
library(jsonlite)
res <- GET("https://ghoapi.azureedge.net/api/Indicator?$filter=contains(IndicatorName,%20%27Death%27)")
data <- res$content
data <- rawToChar(data)
data <- fromJSON(data)
cont <- "Africa"
datasetNUM <- 5
i<-array(c(49,50,51,52 ,53,54,55,57,58,59,60,61,62,63,64,65,66 ,74))
causeName <- data[["value"]]$IndicatorName[i[datasetNUM]]
df<- Final_data_corrected_2[[datasetNUM]]
dfsub <- subset(df,df$continent==cont)
counts <- c("Angola","Burundi","Benin")
library(dplyr)
library(ggplot2)
dfsubsub <- dfsub %>% filter(country %in% counts)
g <- ggplot(dfsubsub,aes(x = country,y = Death_Rate,fill = gender))
g <- g + geom_bar(position = "dodge",stat="identity") + labs(x = "Countries", y = "Death Rate", title = causeName) 
g + theme_new
theme_new=theme(plot.title= element_text(size=16, 
                                         hjust=0.5,
                                         face= "bold"),
                legend.title = element_text(size=14,
                                            face="bold.italic"),
                axis.title = element_text(face="bold"),
                axis.text = element_text(face= "bold"))
