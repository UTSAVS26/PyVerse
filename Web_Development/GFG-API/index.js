const express=require('express');
const userData = require('./userData');

const app = express();

app.get('/',(req,res)=>{
    res.send("GFG API Listening || By Shariq");
})

app.get('/:userName', async (req, res) => {
    try{
        const userName = req.params.userName;
        console.log(userName);
        const data = await userData(userName);
        console.log(data)
        if(!data){
            res.send("User name not found !")
        }
        else{
            res.send(data);
        }
    }
    catch{
        console.error('Error fetching user data');
        res.status(500).send('Internal Server Error');
    }
    // res.send("GFG API Listening || By Shariq");
})

app.listen(3000,()=>{
    console.log("GFG API Listening on PORT 3000");
});