import Axios from "axios";
import Auth from "./auth";

const baseUrl = process.env.REACT_APP_SERVER_API_URL;
// const baseUrl = 'http://127.0.0.1:8000';

const instance = Axios.create({
    baseURL: baseUrl,
    responseType: "json",
});

instance.interceptors.request.use(
    async (config) => {
        let token = sessionStorage.getItem("token");
        if (token) {
            config.headers["Authorization"] = "Bearer " + token;
        }
        return config;
    },
    (error) => {
        Promise.reject(error);
    }
);

instance.interceptors.response.use(
    (response) => {
        return response;
    },
    async (error) => {
        const originalRequest = error.config;
        if (error.response && error.response.status === 401 && originalRequest.url === "/token") {
            return Promise.reject(error);
        }
        if (error.response && error.response.status === 401 && !originalRequest._retry) {
            const refresh_token = await Auth.getRefreshToken();
            if (refresh_token === null) {
                return Promise.reject("No refresh token present");
            }
            originalRequest._retry = true;
            try {
                let { data, status } = await Axios.post(
                    originalRequest.baseURL + "/token",
                    { refresh_token }
                );
                if (status === 200) {
                    Auth.saveAuthorizationToken(data.access_token);
                    Auth.saveRefreshToken(data.refresh_token);
                    instance.defaults.headers["Authorization"] = "Bearer " + data.access_token;
                    return instance(originalRequest);
                }
            } catch (err) {
                return Promise.reject(err);
            }
        }
        return Promise.reject(error);
    }
);

const get = async (url, object) => {
    try {
        const { data, status } = await instance.get(url, {
            headers: object,
        });
        if (status === 200) {
            return data;
        } else {
            throw Error("GET-Request:: Bad Response", status, data);
        }
    } catch (err) {
        throw Error("GET-Request::", err);
    }
};

const post = async (url, object, config = {}) => {
    try {
        const { data, status } = await instance.post(url, object, {
            headers: {
                Authorization: `Bearer ${sessionStorage.getItem('token')}`,
                ...config.headers,
            },
            ...config,
        });
        if (status === 200) {
            return data;
        } else {
            throw Error("POST-Request:: Bad Response", status, data);
        }
    } catch (err) {
        throw err;
    }
};

const put = async (url, object) => {
    try {
        const { data, status } = await instance.put(url, object, {
            headers: {
                Authorization: `Bearer ${sessionStorage.getItem('token')}`,
            },
        });
        if (status === 200) {
            return data;
        } else {
            throw Error("PUT-Request:: Bad Response", status, data);
        }
    } catch (err) {
        throw Error("PUT-Request::", err);
    }
};

const del = async (url) => {
    try {
        const { data, status } = await instance.delete(url, {
            headers: {
                Authorization: `Bearer ${sessionStorage.getItem('token')}`,
            },
        });
        if (status === 200) {
            return data;
        } else {
            throw Error("DELETE-Request:: Bad Response", status, data);
        }
    } catch (err) {
        throw Error("DELETE-Request::", err);
    }
};

const AxiosApi = {
    baseUrl,
    instance,
    get,
    post,
    put,
    del,
};

export default AxiosApi;
