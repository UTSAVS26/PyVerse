df1 <- read.csv("Developed countries gdp.csv")
df2 <- read.csv("Developing countries gdp.csv")
save(df1,file="Developed_countries_gdp.Rdata")
save(df2,file="Developing_countries_gdp.Rdata")
