const puppeteer = require("puppeteer");

async function userData(userName){
    console.log("Fetching...");
    const browser = await puppeteer.launch({
        args: ['--no-sandbox', '--disable-setuid-sandbox']
    });
    const page = await browser.newPage();
    await page.goto(`https://www.geeksforgeeks.org/user/${userName}/`);

    const selector = ".profilePicSection_head_userHandleAndFollowBtnContainer_userHandle__p7sDO";
    try{
        await page.waitForSelector(selector, { timeout: 5000 });
    }
    catch{
        await browser.close();
        return {
            "Status": `User ${userName} not found`,
            "Shariq ": "You have entered wrong username, try again. Thank you!"
        };
    }

    const getUserData = await page.evaluate(() => {
        const API_auther =
            "Shariq https://www.linkedin.com/in/shariq-sd/";
        const name = document.querySelector('.profilePicSection_head_userHandleAndFollowBtnContainer_userHandle__p7sDO').textContent;
        const rank = document.querySelector('.profilePicSection_head_userRankContainer_rank__abngM').firstChild.textContent;
        const institute = document.querySelector('.educationDetails_head_left--text__tgi9I').textContent;
        const overall_coding_score = document.getElementsByClassName("scoreCard_head_card_left--score__pC6ZA")[0].textContent;
        const total_problems_solved = document.getElementsByClassName("scoreCard_head_card_left--score__pC6ZA")[1].textContent;
        const monthly_coding_score = document.getElementsByClassName("scoreCard_head_card_left--score__pC6ZA")[2].textContent;
        const current_streak = document.querySelector(".circularProgressBar_head_mid__IKjUN").childNodes[1].childNodes[0].textContent;
        const languages = document.querySelector('.educationDetails_head_right--text__lLOHI').textContent;

        const problems = document.querySelector('.problemListSection_head__JAiP6');
        let All_solved_problems = [];
        // const school = [];
        // const basic = [];
        // const easy = [];
        // const medium = [];
        // const hard = [];

        // let school_Count = parseInt(document.querySelector('.problemNavbar_head__cKSRi').childNodes[0].firstChild.textContent[8]);
        // let basic_Count = parseInt(document.querySelector('.problemNavbar_head__cKSRi').childNodes[1].firstChild.textContent[7]);
        // let easy_Count = parseInt(document.querySelector('.problemNavbar_head__cKSRi').childNodes[2].firstChild.textContent[6]);
        // let medium_Count = parseInt(document.querySelector('.problemNavbar_head__cKSRi').childNodes[3].firstChild.textContent[8]);
        // let hard_Count = parseInt(document.querySelector('.problemNavbar_head__cKSRi').childNodes[4].firstChild.textContent[6]);

        // if (problems.childNodes[4] != undefined){
        //     school = Array.from(problems.childNodes[4].childNodes[0].childNodes).map((li) => {
        //         return li.childNodes[0].innerText;
        //     })
        // }
        // if (problems.childNodes[3] != undefined) {
        //     basic = Array.from(problems.childNodes[3].childNodes[0].childNodes).map((li) => {
        //         return li.childNodes[0].innerText;
        //     })
        // } 
        // if (problems.childNodes[0] != undefined) {
        //     easy = Array.from(problems.childNodes[0].childNodes[0].childNodes).map((li) => {
        //         return li.childNodes[0].innerText;
        //     })
        // }
        // if (problems.childNodes[1] != undefined) {
        //     medium = Array.from(problems.childNodes[1].childNodes[0].childNodes).map((li) => {
        //         return li.childNodes[0].innerText;
        //     })
        // }

        // if (problems.childNodes[2] != undefined) {
        //     hard = Array.from(problems.childNodes[2].childNodes[0].childNodes).map((li) => {
        //         return li.childNodes[0].innerText;
        //     })
        // }
        for(let i=0;i<problems.childNodes.length;i++){
            All_solved_problems = All_solved_problems.concat(Array.from(problems.childNodes[i].childNodes[0].childNodes).map((li) => {
                return li.childNodes[0].innerText;
            }))
        }
        return {
            API_auther,
            name,
            rank,
            institute,
            overall_coding_score,
            total_problems_solved,
            monthly_coding_score,
            current_streak,
            languages,
            // school,
            // basic,
            // easy,
            // medium,
            // hard
            All_solved_problems
        };
    });
    await browser.close();
    console.log("Fetched");
    return getUserData;
}

module.exports = userData;