library(httr)
library(jsonlite)
library(ggpubr)
res <- GET("https://ghoapi.azureedge.net/api/Indicator?$filter=contains(IndicatorName,%20%27Death%27)")
data <- res$content
data <- rawToChar(data)
data <- fromJSON(data)

(do.call(rbind,Final_data_corrected_2))

newdf <- do.call(rbind,Final_data_corrected_2)


newdf1 <- subset(newdf,country == "Andorra")
i<-array(c(49,50,51,52 ,53,54,55,57,58,59,60,61,62,63,64,65,66 ,74))
causeName <- data[["value"]]$IndicatorName[i]
d <- as.data.frame(rep(causeName,each = 3),nrow = 54)

new_df <- cbind(newdf1,d)


colnames(new_df)[6] <- "Cause_Name"
new_df$gender <- factor(new_df$gender)
new_df$Cause_Name <- factor(new_df$Cause_Name) 


g1 = ggplot(data=new_df, aes(x=" ", y=Death_Rate, group=Cause_Name, colour=Cause_Name, fill=Cause_Name)) +
  geom_bar(data = new_df[new_df$gender == 'BTSX',],width = 1, stat = "identity", col = 1)+
  coord_polar("y", start=0) +theme_void()+
  theme(legend.position = 'none')+
  labs(title = 'BTSX')+
  theme(plot.title = element_text(hjust = 0.5, face = 'bold'))

g2 = ggplot(data=new_df, aes(x=" ", y=Death_Rate, group=Cause_Name, colour=Cause_Name, fill=Cause_Name)) +
  geom_bar(data = new_df[new_df$gender == 'FMLE',],width = 1, stat = "identity", col = 1)+
  coord_polar("y", start=0) +theme_void()+
  theme(legend.position = 'none')+
  labs(title = 'FMLE')+
  theme(plot.title = element_text(hjust = 0.5, face = 'bold'))
  

g3 = ggplot(data=new_df, aes(x=" ", y=Death_Rate, group=Cause_Name, colour=Cause_Name, fill=Cause_Name)) +
  geom_bar(data = new_df[new_df$gender == 'MLE',],width = 1, stat = "identity", col = 1)+
  coord_polar("y", start=0) +theme_void()+
  theme(legend.position = 'none')+
  labs(title = 'MLE')+
  theme(plot.title = element_text(hjust = 0.5, face = 'bold'))


ggarrange(g1,g2,g3, common.legend = TRUE, legend="right", nrow = 1)+
  plot_annotation(title = 'Piechart for different genders',
                  theme = theme(plot.title = element_text(size = 18,face = 'bold'),
                                legend.title = element_text(face = 'bold')))

 
  