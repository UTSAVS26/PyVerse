let _identity = undefined;
let authenticated = false;

function isAuthenticated() {
    return authenticated;
}

function isIdentityResolved() {
    return typeof _identity != "undefined";
}

function getIdentity() {
    return _identity;
}

function setIdentity(identity) {
    _identity = identity;
    authenticated = typeof identity != "undefined";
}

function clear() {
    authenticated = false;
    _identity = undefined;
}

const PrincipalState = {
    isAuthenticated,
    isIdentityResolved,
    getIdentity,
    setIdentity,
    clear,
};

export default PrincipalState;
