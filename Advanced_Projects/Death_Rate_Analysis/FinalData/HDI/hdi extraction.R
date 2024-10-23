library(tidyverse)
library(rvest)
html <- read_html("https://countryeconomy.com/hdi?year=2004")
table <- html %>% html_table()
df <- table[[1]]
df
dim(df)
write.csv(df,"hdi.csv",row.names=FALSE)
df1 <- read.csv("hdi.csv")
colnames(df1) <- c("HDI", "Country.Codes")
save(df1 ,file = "HDI.Rdata")
