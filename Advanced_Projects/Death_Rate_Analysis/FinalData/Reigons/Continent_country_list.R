load("gender+country/Final_data_3.Rdata")
CnC <- Final_data_corrected_2[[1]]
CnC <- CnC[,4:5]
Africa <- c(unique(subset(CnC,CnC$continent=="Africa"))[,1])
Americas <- c(unique(subset(CnC,CnC$continent=="Americas"))[,1])
Eastern <- c(unique(subset(CnC,CnC$continent=="Eastern Mediterranean"))[,1])
Europe <- c(unique(subset(CnC,CnC$continent=="Europe"))[,1])
SE <- c(unique(subset(CnC,CnC$continent=="South-East Asia"))[,1])
Western <- c(unique(subset(CnC,CnC$continent=="Western Pacific"))[,1])
listcc <- list(Africa,Americas,Eastern,Europe,SE,Western)
save(listcc, file = "Plots/ListOfCountriesContinentWise.Rdata")
