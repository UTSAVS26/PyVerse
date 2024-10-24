library(httr)
library(jsonlite)
res <- GET("https://ghoapi.azureedge.net/api/Indicator?$filter=contains(IndicatorName,%20%27Death%27)")
data <- res$content
data <- rawToChar(data)
data <- fromJSON(data)
country <- "Andorra"
#newdf <- matrix(nrow = 8154,ncol = 5)
(do.call(rbind,Final_data_corrected_2))

newdf <- do.call(rbind,Final_data_corrected_2)

newdf1 <- subset(newdf,country == "Angola")


newdf2 <- newdf1[,-2]
newdf3 <- data.frame(matrix(nrow=18,ncol=4))

for(i in 1:18){
  newdf3[i,2] <- newdf2[i*3-2,2] + newdf2[i*3-1,2] + newdf2[i*3,2]
}
newdf3[,1] = newdf2[1:18,1]
newdf3[,3] = newdf2[1:18,3]
newdf3[,4] = newdf2[1:18,4]
colnames(newdf3) <- colnames(newdf2)
newdf3

i<-array(c(49,50,51,52 ,53,54,55,57,58,59,60,61,62,63,64,65,66 ,74))
causeName <- data[["value"]]$IndicatorName[i]
d <- as.data.frame(causeName,nrow = 18)
new_df <- cbind(newdf3,d)
library(plotly)
fig <- plot_ly(new_df,labels = causeName, values = new_df$Death_Rate,type = 'pie')
fig <- fig %>% layout(title = 'Different Type of Death Causes of Angola ',
                      xaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
                      yaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE))

fig





