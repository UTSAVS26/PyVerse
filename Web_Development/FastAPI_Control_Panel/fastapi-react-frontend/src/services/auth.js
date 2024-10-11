import PrincipalService from "./principal.service";
// import { useNavigate } from "react-router-dom";

let _authToken = undefined;
let _refreshToken = undefined;

async function authorize(force) {
    // const navigate=useNavigate();
    try {
        await PrincipalService.identity(force);
        const isAuthenticated = PrincipalService.isAuthenticated();
        const location = window.location;
        if (isAuthenticated && (location.pathname === "/login" || location.pathname === "/register")) {
            // navigate("/dashboard");
        } else if (!isAuthenticated && location.pathname !== "/login") {
            storePreviousState(location);
            // navigate("/login");
        }
    } catch (err) {
        console.log("Failed to get identity", err);
        // navigate("/login")
    }
}


function cleanAuth() {
    // saveAuthorizationToken();
    // saveRefreshToken();
    PrincipalService.clear();
}

async function getAuthorizationToken() {
    if (_authToken === undefined) _authToken = await get("token");
    // _authToken = sessionStorage.getItem("auth");
    return _authToken;
}

function saveAuthorizationToken(authToken) {
    save("token", authToken);
    // sessionStorage.setItem("auth", authToken);
    _authToken = authToken;
}

async function getRefreshToken() {
    if (_refreshToken === undefined) _refreshToken = await get("refresh");
    return _refreshToken;
}

function saveRefreshToken(refreshToken) {
    save("refresh", refreshToken);
    // sessionStorage.setItem("refresh", refreshToken);
    _refreshToken = refreshToken;
}

async function getPreviousState() {
    // return JSON.parse(sessionStorage.getItem("state"));
    return await JSON.parse(get("state"));
}

function resetPreviousState() {
    // sessionStorage.removeItem("state");
    remove("state");
}

function storePreviousState(state) {
    // sessionStorage.setItem("state", JSON.stringify(state));
    save("state", JSON.stringify(state));
}

const save = async (key, value) => {
    await sessionStorage.setItem(key,value);
};

const get = async (key) => {
    return sessionStorage.getItem(key);
};

const remove = async (key) => {
    await sessionStorage.removeItem(key);
}
const Auth = {
    authorize,
    getPreviousState,
    resetPreviousState,
    getAuthorizationToken,
    saveAuthorizationToken,
    getRefreshToken,
    saveRefreshToken,
    cleanAuth,
};

export default Auth;
