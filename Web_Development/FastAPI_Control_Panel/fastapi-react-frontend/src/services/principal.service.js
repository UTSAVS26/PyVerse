import AxiosApi from "./axios.api";
import PrincipalState from "./principal.state";

async function identity(force) {
    if (force === true) PrincipalState.setIdentity(undefined);
    if (typeof PrincipalState.getIdentity() != "undefined")
        return PrincipalState.getIdentity();
    try {
        const data = await AxiosApi.get("/users/me");
        PrincipalState.setIdentity(data);
        return data;
    } catch (err) {
        return PrincipalState.setIdentity();
    }
}

function isAuthenticated() {
    return PrincipalState.isAuthenticated();
}

function isIdentityResolved() {
    return PrincipalState.isIdentityResolved();
}

function clear() {
    PrincipalState.clear();
}

function hasAnyRole(role) {
    let _identity = PrincipalState.getIdentity();
    if (!PrincipalState.isAuthenticated() || !_identity || !_identity.roles)
        return false;
    return _identity.role===role;
}

function isAdmin() {
    return hasAnyRole(["ADMIN"]);
}

function isNormalUser() {
    return hasAnyRole(["USER"]);
}

function isSystemAdmin() {
    return hasAnyRole(["SYSTEM_ADMIN"]);
}

function getCurrentUser() {
    return PrincipalState.getIdentity();
}

const PrincipalService = {
    identity,
    isAuthenticated,
    isIdentityResolved,
    isAdmin,
    isSystemAdmin,
    isNormalUser,
    clear,
    getCurrentUser,
};

export default PrincipalService;
