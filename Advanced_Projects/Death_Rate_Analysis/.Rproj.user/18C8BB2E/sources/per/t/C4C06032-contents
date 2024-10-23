#install.packages("plotly")
#install.packages("gridExtra")
#install.packages("dplyr")
#install.packages("ggplot2")
#install.packages("ggpubr")
#install.packages("patchwork")
all_countries<-c(listcc[[1]],listcc[[2]],listcc[[3]],listcc[[4]],listcc[[5]],listcc[[6]])
if (interactive()){
  library(shiny)
  # Define UI for application that draws a histogram
  ui <- fluidPage(
    
    # Application title
    titlePanel("Causes of Deaths"),
    
    # Sidebar with a slider input for number of bins 
    sidebarLayout(
      sidebarPanel(
        selectInput("plotType",
                    "What type of Data you want to visualize?",
                    c("Choose from the List"="nothing1","Death Rates vs Countries"="select_disease","Gender Wise MultiBar Plots"="gender_multibar","Pie Chart of Different Causes"="pie","Analysis with respect to HDI"="ahdi","Analysis with respect to GDP"="agdp")),
        conditionalPanel(
          condition = "input.plotType == 'select_disease'",
          selectInput(
            "dis", "Cause:",
            c("Choose the Cause"=0,"Alcohol"=1, "Breast Cancer"=2, "Colon and Rectum Cancer"=3, "Diabetes" = 4,"Drowings"=5,"Falls"=6,"Fires"=7,"Liver Cancer"=8,"Liver Cirrhosis"=9,"Mouth And Oropharynx Cancer"=10,"Oesophagus"=11,"Poisioning"=12,"Prematurity and Low Birth Rate"=13,"Road Traffic Accidents"=14,"Self Inflicted Injury"=15,"Other Unintentional Injuries"=16,"Violence"=17,"Cerebrovascular Diseases"=18),selected=0
          )
        ), conditionalPanel(
          condition = "input.plotType == 'gender_multibar'",
          selectInput(
            "dis1", "Cause:",
            c("Choose the Cause"=0,"Alcohol"=1, "Breast Cancer"=2, "Colon and Rectum Cancer"=3, "Diabetes" = 4,"Drowings"=5,"Falls"=6,"Fires"=7,"Liver Cancer"=8,"Liver Cirrhosis"=9,"Mouth And Oropharynx Cancer"=10,"Oesophagus"=11,"Poisioning"=12,"Prematurity and Low Birth Rate"=13,"Road Traffic Accidents"=14,"Self Inflicted Injury"=15,"Other Unintentional Injuries"=16,"Violence"=17,"Cerebrovascular Diseases"=18),selected=0
          ),
          
          selectInput("cont","Select Region:",
                      c("Choose the Continent"="nothing3","Africa","Americas","Eastern Mediterranean","Europe","South-East Asia","Western Pacific")),
          conditionalPanel(
            condition = "input.cont == 'Africa'",
            checkboxGroupInput("checkGroup", 
                               "Select Countries:", 
                               choices = listcc[[1]])),
          conditionalPanel(
            condition = "input.cont == 'Americas'",
            checkboxGroupInput("checkGroup1", 
                               "Select Countries:", 
                               choices =  listcc[[2]])),
          conditionalPanel(
            condition = "input.cont == 'Eastern Mediterranean'",
            checkboxGroupInput("checkGroup2", 
                               "Select Countries:", 
                               choices =  listcc[[3]])),
          conditionalPanel(
            condition = "input.cont == 'Europe'",
            checkboxGroupInput("checkGroup3", 
                               "Select Countries:", 
                               choices =  listcc[[4]])),
          conditionalPanel(
            condition = "input.cont == 'South-East Asia'",
            checkboxGroupInput("checkGroup4", 
                               "Select Countries:", 
                               choices =  listcc[[5]])),
          conditionalPanel(
            condition = "input.cont == 'Western Pacific'",
            checkboxGroupInput("checkGroup5", 
                               "Select Countries:", 
                               choices =  listcc[[6]]))
          
          
        ), 
        conditionalPanel(
          condition = "input.plotType ==  'pie'",
          selectInput(
            "count", "Country",
            c("Choose The Country"="nothing4",all_countries)
          ),
          checkboxGroupInput("gend",
                             "Choose Gender:",
                             choices=c("All","Female","Male","Others"))
        ),
        conditionalPanel(
          condition = "input.plotType == 'ahdi'",
          selectInput("cor","Type Of Correlation:",
                      choices = c("Choose a Correlation"="nothing5","Positive Correlation"="pos","Negative Correlation"="neg","No Correlation"="nocor")),
          
          conditionalPanel(
            condition = "input.cor == 'neg' ",
            selectInput("dis5","Choose a Cause:",
                        choices = c("Choose from the List"="nothing6","Diabetes Mellitus"=4,"Drownings"=5,"Fires"=7,"Liver Cancer"=8,"Oesophagus cancer"=11,"Prematurity"=13,"Road traffic accidents"=14,"Violence"=17,"CerebroVascular"=18)
            )
          ),
          conditionalPanel(
            condition = "input.cor == 'pos' ",
            selectInput("dis6","Choose a Cause:",
                        choices = c("Choose from the List"="nothing7","Colon rectum cancer"=3)
            )
          ),
          conditionalPanel(
            condition = "input.cor == 'nocor' ",
            selectInput("dis7","Choose a Cause:",
                        choices = c("Choose from the List"="nothing8","Alcohol"=1,"Breast cancer"=2,"Falls"=6,"Liver Cirrhosis"=9,"Poisoning"=12,"Self-inflicted"=15)
            )
          )
        ),
        conditionalPanel(
          condition = "input.plotType == 'agdp'",
          selectInput("cor1","Type Of Correlation:",
                      choices = c("Choose a Correlation"="nothing9","Positive Correlation"="pos1","Negative Correlation"="neg1","No Correlation"="nocor1")),
          
          conditionalPanel(
            condition = "input.cor1 == 'neg1' ",
            selectInput("dis8","Choose a Cause:",
                        choices = c("Choose from the List"="nothing10","Diabetes Mellitus"=52,"Drownings"=53,"Fires"=55,"Liver Cancer"=57,"Mouth and Oropharynx"=59,"Oesophagus cancer"=60,"Prematurity"=62,"Road traffic accidents"=63,"Violence"=66,"CerebroVascular"=74)
            )
          ),
          conditionalPanel(
            condition = "input.cor1 == 'pos1' ",
            selectInput("dis9","Choose a Cause:",
                        choices = c("Choose from the List"="nothing11","Colon rectum cancer"=51)
            )
          ),
          conditionalPanel(
            condition = "input.cor1 == 'nocor1' ",
            selectInput("dis10","Choose a Cause:",
                        choices = c("Choose from the List"="nothing12","Alcohol"=49,"Breast cancer"=50,"Falls"=54,"Liver Cirrhosis"=58,"Poisoning"=61,"Self-inflicted"=64,"Unintentional"=65)
            )
          )
        )
      ),
      # Show a plot of the generated distribution
      mainPanel(
        plotOutput("plot1"),
        plotOutput("plot2"),
        plotlyOutput("piechart"),
        #plotOutput("plot3")
        textOutput("bottom")
      )
    )
    
  )
  
  # Define server logic required to draw a histogram
  server <- function(input, output) {
    
    output$plot1 <- renderPlot({
      if (input$plotType == "select_disease"){
        if(input$dis!="0"){
          
          
          ###
          #load("gender+country/Final_data_3.Rdata")
          df<-Final_data_corrected_2[[as.numeric(input$dis)]] ###
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
          male<-which(df$gender=="Male")    
          #df
          
          g1 <- ggplot(df[deve,],aes(x=index,y=Death_Rate,color=gender))+geom_point()+coord_cartesian(ylim = c(0, limit))
          #g1<-g1+stat_smooth(formula = y~x,method="lm",data=df[deve,],aes(group=1,col=gender))
          g1 <- g1+ theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1),legend.title = element_text(face="bold"))+xlab("Developed Countries")
          g1<-g1 + scale_color_manual(labels = c("Others", "Female","Male"),values=c("red","green","blue"))
          g1<-g1 ##+labs(title = "Cause\n")
          
          for(i in 1:length(developing$Country.Codes)){
            x<-which(df$index==developing$Country.Codes[i])
            if(length(x)!=0){
              deving<-append(deving,x)
            }
          }
          
          g2 <- ggplot(df[deving,],aes(x=index,y=Death_Rate,color=gender,ylim=limit))+geom_point()+coord_cartesian(ylim = c(0, limit))
          # g2<-g2+stat_smooth(formula = y~x,method="lm",aes(group=1))
          
          g2 <- g2+ theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1),legend.title = element_text(face="bold"))+xlab("Developing Countries")
          g2<-g2 + scale_color_manual(labels = c("Others", "Female","Male"),values=c("red","green","blue"))
          g2<-g2 #+guides(fill=guide_legend(title="Gender"))
          grid.arrange(g1, g2, ncol = 2)
          
          
          ###
          ####
        }
      }
      else if (input$plotType == "gender_multibar"){
        #if(as.numeric(input$dis) != 0){
        
        if(input$cont != "nothing3"){
          cont <- input$cont ###
          datasetNUM <- as.numeric(input$dis1) ###
          if(datasetNUM != 0){
            
            i<-array(c(49,50,51,52 ,53,54,55,57,58,59,60,61,62,63,64,65,66,74))
            causeName <- data[["value"]]$IndicatorName[i[datasetNUM]]
            df<- Final_data_corrected_2[[datasetNUM]]
            dfsub <- subset(df,df$continent==cont)
            ###
            if(cont == "Africa"){
              counts <- unlist(input$checkGroup)
            } else if(cont == "Americas"){
              counts <- unlist(input$checkGroup1)
            } else if(cont == "Eastern Mediterranean"){
              counts <- unlist(input$checkGroup2)
            }
            else if(cont == "Europe"){
              counts <- unlist(input$checkGroup3)
            }
            else if(cont == "South-East Asia"){
              counts <- unlist(input$checkGroup4)
            }
            else if(cont == "Western Pacific"){
              counts <- unlist(input$checkGroup5)
            }
            
            dfsubsub <- dfsub %>% filter(country %in% counts)
            g <- ggplot(dfsubsub,aes(x = country,y = Death_Rate,fill = gender))
            g <- g + geom_bar(position = "dodge",stat="identity") + labs(x = "Countries", y = "Death Rate", title = causeName) 
            g<-g + theme(plot.title= element_text(size=16, 
                                               hjust=0.5,
                                               face= "bold"),
                      legend.title = element_text(size=14,
                                                  face="bold.italic"),
                      axis.title = element_text(face="bold"),
                      axis.text = element_text(face= "bold"))
            g+scale_fill_discrete(labels = c("Others", "Female","Male"))
          }
        }
      }
      else if(input$plotType=="pie"){
        if(length(which(input$gend == "Male"))!=0){
          
          (do.call(rbind,Final_data_corrected_2))
          
          newdf <- do.call(rbind,Final_data_corrected_2)
          
          
          newdf1 <- subset(newdf,country == input$count)
          nrow(newdf1)
          i<-array(c(49,50,51,52 ,53,54,55,57,58,59,60,61,62,63,64,65,66 ,74))
          causeName <- data[["value"]]$IndicatorName[i]
          d <- as.data.frame(rep(causeName,each = 3),nrow = 54)
          
          new_df <- cbind(newdf1,d)
          
          
          colnames(new_df)[6] <- "Cause_Name"
          new_df$gender <- factor(new_df$gender)
          new_df$Cause_Name <- factor(new_df$Cause_Name) 
          
          
          g3 = ggplot(data=new_df, aes(x=" ", y=Death_Rate, group=Cause_Name, colour=Cause_Name, fill=Cause_Name)) +
            geom_bar(data = new_df[new_df$gender == 'MLE',],width = 1, stat = "identity", col = 1)+
            coord_polar("y", start=0) +theme_void()+
            theme(legend.position = 'none')+
            labs(title = 'Male')+
            theme(plot.title = element_text(hjust = 0.5, face = 'bold'))
          
          
          ggarrange(g3 ,common.legend = TRUE, legend="right", nrow = 1)+
            plot_annotation(title = 'Piechart for different genders',
                            theme = theme(plot.title = element_text(size = 18,face = 'bold'),
                                          legend.title = element_text(face = 'bold')))
        }
      } else if(input$plotType == "ahdi"){
        if(input$cor!="nothing5"){
          if((input$cor=="neg" && input$dis5 != "nothing6") || (input$cor=="pos" && input$dis6 != "nothing7") ||(input$cor=="nocor" && input$dis7 != "nothing8")){
            ##datasetNUM <- as.numeric(input$dis) ###
            
            if(input$cor =="neg"){
              datasetNUM <- as.numeric(input$dis5)
            } else if(input$cor =="pos"){
              datasetNUM <- as.numeric(input$dis6)
            } else if(input$cor =="nocor"){
              datasetNUM <- as.numeric(input$dis7)
            }
            #i<-array(c(49,50,51,52 ,53,54,55,57,58,59,60,61,62,63,64,65,66 ,74))
            df<-Final_data_corrected_2[[datasetNUM]]
            newdf<-data.frame(matrix(nrow=length(HDI$HDI),ncol=5))
            colnames(newdf)<-c("cc","Death_rate","HDI","country","continent")
            newr<-1
            colnames(HDI) <- c("Country.Codes","HDI")
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
            #g1 <- ggplot(newdf,aes(x=factor(cc,levels = cc[order(HDI)]),y=Death_rate))+geom_point(color=1)+theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
            
            g2 <- ggplot(newdf,aes(x=factor(cc,levels = cc[order(HDI)]),y=Death_rate))+geom_point(aes(color=1))+theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1),legend.position = "none")+labs(x="Countries (Arranged in Increasing Order of HDI)")
            g2<-g2+stat_smooth(formula = y~x,method="lm",aes(group=1))
            g2 <- g2 +geom_line(aes(y=HDI/scale,color=2,group=1))+scale_y_continuous(sec.axis = sec_axis(~.*scale,name = "HDI (Represented by Light Blue Line)"))
            g2<-g2+ theme(plot.title= element_text(size=16, 
                                                     hjust=0.5,
                                                     face= "bold"),
                            legend.title = element_text(size=14,
                                                        face="bold.italic"),
                            axis.title = element_text(face="bold"),
                            axis.text = element_text(face= "bold"))
            g2
            
            
            
          }
        }
      } else if(input$plotType == "agdp"){
        if(input$cor1!="nothing5"){
          if((input$cor1=="neg1" && input$dis8 != "nothing10") || (input$cor1=="pos1" && input$dis9 != "nothing11") ||(input$cor1=="nocor1" && input$dis10 != "nothing12")){
            developing_gdp <- developing_gdp[,c(1,2,4,6)]
            colnames(developing_gdp)<-colnames(developed_gdp)
            GDP <- rbind(developed_gdp,developing_gdp)
            GDP <- GDP[,3:4]
            ccid<-array(c(49,50,51,52,53,54,55,57,58,59,60,61,62,63,64,65,66 ,74))
            
            
            if(input$cor1 =="neg1"){
              datasetNUM <- as.numeric(input$dis8)
            } else if(input$cor1 =="pos1"){
              datasetNUM <- as.numeric(input$dis9)
            } else if(input$cor1 =="nocor1"){
              datasetNUM <- as.numeric(input$dis10)
            }
            
            df<-Final_data_corrected_2[[match(datasetNUM,ccid)]]
            newdf<-data.frame(matrix(nrow=length(GDP$GDP),ncol=5))
            colnames(newdf)<-c("cc","Death_rate","GDP","country","continent")
            newr<-1
            for(x in 1:length(GDP$Country.Codes))
            {
              r <- which(df$index==GDP$Country.Codes[x])
              
              if(length(r)!=0)
              {
                newdf[newr,1]<-df$index[r[1]]
                newdf[newr,2]<-sum(df$Death_Rate[r])
                newdf[newr,3]<-GDP$GDP[x]/100000
                newdf[newr,4]<-df$country[r[1]]
                newdf[newr,5]<-df$continent[r[1]]
                newr<-newr+1
              }
              
            }
            
            
            newdf<-na.omit(newdf)
            maxd<- max(newdf$Death_rate)
            scale <- 1/maxd
            g20 <- ggplot(newdf,aes(x=factor(cc,levels = cc[order(GDP)]),y=Death_rate))+geom_point(aes(color=1))+theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1),legend.position = "none")+labs(x="Countries (Arranged in Increasing Order of GDP)")
            g20 <- g20+ stat_smooth(formula=y~x,method="lm",aes(group=1))
            g20 <- g20 +geom_line(aes(y=GDP/scale,color=2,group=1))+scale_y_continuous(sec.axis = sec_axis(~.*scale,name = "GDP per capita /10e5"))
            g20<-g20+ theme(plot.title= element_text(size=16, 
                                                   hjust=0.5,
                                                   face= "bold"),
                          legend.title = element_text(size=14,
                                                      face="bold.italic"),
                          axis.title = element_text(face="bold"),
                          axis.text = element_text(face= "bold"))
            g20
            
            
            
            
            
          }
        }
      }
      #}
    }) 
    
    output$plot2<- renderPlot({### ALong with HDI
      if(input$plotType=="pie"){
        if(length(which(input$gend == "Female"))!=0 && length(which(input$gend == "Others"))!=0){
          
          (do.call(rbind,Final_data_corrected_2))
          
          newdf <- do.call(rbind,Final_data_corrected_2)
          
          
          newdf1 <- subset(newdf,country == input$count)
          nrow(newdf1)
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
            labs(title = 'Others')+
            theme(plot.title = element_text(hjust = 0.5, face = 'bold'))
          
          g2 = ggplot(data=new_df, aes(x=" ", y=Death_Rate, group=Cause_Name, colour=Cause_Name, fill=Cause_Name)) +
            geom_bar(data = new_df[new_df$gender == 'FMLE',],width = 1, stat = "identity", col = 1)+
            coord_polar("y", start=0) +theme_void()+
            theme(legend.position = 'none')+
            labs(title = 'Female')+
            theme(plot.title = element_text(hjust = 0.5, face = 'bold'))
          
          
          
          ggarrange(g1,g2 ,common.legend = TRUE, legend="right", nrow = 1)+
            plot_annotation(
                            theme = theme(plot.title = element_text(size = 18,face = 'bold'),
                                          legend.title = element_text(face = 'bold')))
        } else if(length(which(input$gend == "Female"))!=0){
          
          
          (do.call(rbind,Final_data_corrected_2))
          
          newdf <- do.call(rbind,Final_data_corrected_2)
          
          
          newdf1 <- subset(newdf,country == input$count)
          nrow(newdf1)
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
            labs(title = 'Female')+
            theme(plot.title = element_text(hjust = 0.5, face = 'bold'))
          
          
          
          ggarrange(g2 ,common.legend = TRUE, legend="right", nrow = 1)+
            plot_annotation(
                            theme = theme(plot.title = element_text(size = 18,face = 'bold'),
                                          legend.title = element_text(face = 'bold')))
        } else if(length(which(input$gend == "Others"))!=0){
          
          
          (do.call(rbind,Final_data_corrected_2))
          
          newdf <- do.call(rbind,Final_data_corrected_2)
          
          
          newdf1 <- subset(newdf,country == input$count)
          nrow(newdf1)
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
            labs(title = 'Others')+
            theme(plot.title = element_text(hjust = 0.5, face = 'bold'))
          
          ggarrange(g1 ,common.legend = TRUE, legend="right", nrow = 1)+
            plot_annotation(
                            theme = theme(plot.title = element_text(size = 18,face = 'bold'),
                                          legend.title = element_text(face = 'bold')))
        } 
      }
    })
    
    output$piechart<-renderPlotly({
      if(input$plotType == 'pie'){
        if(length(which(input$gend == "All"))!=0 ){
          if(input$count!="nothing4"){
            
            country <- "Andorra" #input$dis
            #newdf <- matrix(nrow = 8154,ncol = 5)
            (do.call(rbind,Final_data_corrected_2))
            
            newdf <- do.call(rbind,Final_data_corrected_2)
            
            newdf1 <- subset(newdf,country == input$count)
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
            
            fig <- plot_ly(new_df,labels = causeName, values = new_df$Death_Rate,type = 'pie')
            fig <- fig %>% layout(title = paste0('Different Type of Death Causes of ',input$count),
                                  xaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
                                  yaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE))
            fig
          }}
      }
    })
    output$bottom<-renderText({
      if(input$plotType == 'pie'){
        if(length(which(input$gend == "All"))!=0 ){
          if(input$count!="nothing4"){
            "The Pie Chart For all the Genders is an Interactive plot.
             We can add or remove the diseases we want to visualize in the pie chart by clicking it in the legend."
          }
        }
      }
    })
  }
  
  # Run the application 
  shinyApp(ui = ui, server = server)
}
