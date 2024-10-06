setInterval(()=>{
    var Btn = document.getElementsByClassName("ytp-ad-skip-button");
    if(Btn!=undefined && Btn.length > 0){
        console.log("Skipping Ad");
        Btn[0].click();
    }
},3000);