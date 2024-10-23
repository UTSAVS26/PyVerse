df <- read_csv("Health index.csv")
df <- df[,1:3]
df <- na.omit(df)
colnames(df) <- c("country","country code","Health index")
df[-1,]
save(df,file = "Health_Index,Rdata")
