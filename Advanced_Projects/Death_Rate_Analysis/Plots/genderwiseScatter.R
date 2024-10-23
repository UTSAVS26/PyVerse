library(httr)
library(jsonlite)
res <- GET("https://ghoapi.azureedge.net/api/Indicator?$filter=contains(IndicatorName,%20%27Death%27)")
data <- res$content
data <- rawToChar(data)
data <- fromJSON(data)
library(ggplot2)
load("gender+country/Final_data_3.Rdata")
#df$SpatialDim
#developed$Country.Codes
#developing$Country.Codes
#which(df$SpatialDim==developed$Country.Codes[1])
#which(df$SpatialDim==developing$Country.Codes[1])
ccid<-array(c(54,55,56,57,58,59,61,62,63,64,65,66,67,68,69,70))
df<-Final_data_corrected_2[[16]]
df[is.na(df)]<-0
limit <- max(df$Death_Rate)
deve<-numeric(length = 0)
deving<-numeric(length=0)
for(i in 1:length(developed$Country.Codes)){
  x<-which(df$index==developed$Country.Codes[i])
  if(length(x)!=0){
    deve<-append(deve,x)
  }
}

g1 <- ggplot(df[deve,],aes(x=index,y=Death_Rate,color=gender))+geom_point()+coord_cartesian(ylim = c(0, limit))
g1 <- g1+ theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))+xlab("Developed Countries")

for(i in 1:length(developing$Country.Codes)){
  x<-which(df$index==developing$Country.Codes[i])
  if(length(x)!=0){
    deving<-append(deving,x)
  }
}

g2 <- ggplot(df[deving,],aes(x=index,y=Death_Rate,color=gender,ylim=limit))+geom_point()+coord_cartesian(ylim = c(0, limit))
g2 <- g2+ theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))+xlab("Developing Countries")

library("gridExtra")
grid.arrange(g1, g2, ncol = 2) 

