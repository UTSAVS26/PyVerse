library(httr)
library(jsonlite)
#Using API to extract data from WHO website
res <- GET("https://ghoapi.azureedge.net/api/Indicator?$filter=contains(IndicatorName,%20%27Death%27)")
data <- res$content
data <- rawToChar(data)
data <- fromJSON(data)
load("Country/Accidents.Rdata")
load("Country/Genetic_and_lifestyle.Rdata")
load("Country/Poisoning.Rdata")
load("Country/Substance_abuse.Rdata")
load("gender+country/Final_data_3.Rdata")
load("Cleaned data/countries sorted/developed.Rdata")
load("Cleaned data/countries sorted/developing.Rdata")
HDI <- rbind(developed,developing)
typenames <- c("Substance Abuse","Genetic and Lifestyle","Accidents","Poisoning")
typeNUM <- 3
l <- l3
library(gridExtra)
library(ggplot2)

########################################################################################################################
########################################################################################################################
#iss section mae saare plots ek hi saath plot ho rahe hain



plotlist <- list()
for(i in 1:length(l))
{
  df <- l[[i]]
  newdf<-data.frame(matrix(nrow=length(HDI$HDI),ncol=3))
  colnames(newdf)<-c("cc","Death_rate","HDI")
  newr<-1
  for(x in 1:length(HDI$Country.Codes))
  {
    r <- which(df$mat1..i...SpatialDim==HDI$Country.Codes[x])
    
    if(length(r)!=0)
    {
      newdf[newr,1]<-df$mat1..i...SpatialDim[r]
      newdf[newr,2]<-df$mat1..i...NumericValue[r]
      

      newdf[newr,3]<-HDI$HDI[x]
      newr<-newr+1
    }
    
  }
  
  
  newdf<-na.omit(newdf)
  maxd<- max(newdf$Death_rate)
  scale <- 1/maxd
  g <- ggplot(newdf,aes(x=factor(cc,levels = cc[order(HDI)]),y=Death_rate))+geom_point(aes(color=1))+theme(axis.text.x = element_text(angle = 90, size=5,vjust = 0.5, hjust=1),legend.position = "none")
  g <- g +geom_line(aes(y=HDI/scale,color=2,group=1))+scale_y_continuous(sec.axis = sec_axis(~.*scale,name = "HDI"))+xlab("Countries")
  plotlist[[i]] <- g
  
}
marrangeGrob(plotlist,ncol=2,nrow=2)


############################################################################################################################################
############################################################################################################################################

Lall <- c(l1,l2,l3,l4)
k <- 7
df <- Lall[[k]]
newdf<-data.frame(matrix(nrow=length(HDI$HDI),ncol=3))
colnames(newdf)<-c("cc","Death_rate","HDI")
newr<-1
for(x in 1:length(HDI$Country.Codes))
{
  r <- which(df$mat1..i...SpatialDim==HDI$Country.Codes[x])
  
  if(length(r)!=0)
  {
    newdf[newr,1]<-df$mat1..i...SpatialDim[r]
    newdf[newr,2]<-df$mat1..i...NumericValue[r]
    
    
    newdf[newr,3]<-HDI$HDI[x]
    newr<-newr+1
  }
  
}


newdf<-na.omit(newdf)
maxd<- max(newdf$Death_rate)
scale <- 1/maxd
g <- ggplot(newdf,aes(x=factor(cc,levels = cc[order(HDI)]),y=Death_rate))+geom_point(aes(color=1))+theme(axis.text.x = element_text(angle = 90,size=10,vjust = 0.5, hjust=1),legend.position = "none")
g <- g+ stat_smooth(formula=y~x,method="lm",aes(group=1))
g <- g +geom_line(aes(y=HDI/scale,color=2,group=1))+scale_y_continuous(sec.axis = sec_axis(~.*scale,name = "HDI"))+xlab("Countries")
g

