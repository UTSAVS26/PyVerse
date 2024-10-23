library(httr)
library(jsonlite)
res <- GET("https://ghoapi.azureedge.net/api/Indicator?$filter=contains(IndicatorName,%20%27Death%27)")
data <- res$content
data <- rawToChar(data)
data <- fromJSON(data)
library(ggplot2)
load("gender+country/Final_data_3.Rdata")
load("Cleaned Data/countries sorted/HDI.Rdata")
HDI <- df1
colnames(HDI)<- c("Country.Codes","HDI")
ccid<-array(c(49,50,51,52,53,54,55,57,58,59,60,61,62,63,64,65,66 ,74))
datasetNUM <- 49
df<-Final_data_corrected_2[[match(datasetNUM,ccid)]]
newdf<-data.frame(matrix(nrow=length(HDI$HDI),ncol=5))
colnames(newdf)<-c("cc","Death_rate","HDI","country","continent")
newr<-1
for(x in 1:length(HDI$Country.Codes))
{
  r <- which(df$index==HDI$Country.Codes[x])
  
  if(length(r)!=0)
  {
    newdf[newr,1]<-df$index[r[1]]
    newdf[newr,2]<-sum(df$Death_Rate[r])
    newdf[newr,3]<-HDI$HDI[x]
    newdf[newr,4]<-df$country[r[1]]
    newdf[newr,5]<-df$continent[r[1]]
    newr<-newr+1
  }
  
}


newdf<-na.omit(newdf)
maxd<- max(newdf$Death_rate)
scale <- 1/maxd
g2 <- ggplot(newdf,aes(x=factor(cc,levels = cc[order(HDI)]),y=Death_rate))+geom_point(aes(color=1))+theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1),legend.position = "none")
g2 <- g2+ stat_smooth(formula=y~x,method="lm",aes(group=1))
g2 <- g2 +geom_line(aes(y=HDI/scale,color=2,group=1))+scale_y_continuous(sec.axis = sec_axis(~.*scale,name = "HDI"))
g2


