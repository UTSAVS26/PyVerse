import jwt from 'jsonwebtoken';
const JWT_SECRET = process.env.JWT_SECRET || 'Skanarul$123@';

const fetchuser = (req, res, next) => {
    // Get the user from the jwt token and add id to req object
    const authHeader = req.header('Authorization');
    const token = authHeader && authHeader.startsWith('Bearer ') ? authHeader.split(' ')[1] : null;
    console.log("Received jet secret Token:", JWT_SECRET);
    if(!token) {
        res.status(401).send({error: "Please authenticate using a valid token"});
    }
    try {
        console.log("before Decoded Data:");
        const data = jwt.verify(token, JWT_SECRET);
        console.log("Decoded Data:", data);
        req.user = data.user;
        next();
    } catch(error) {
        console.error("Token verification failed:", error.name, error.message);
        res.status(401).send({error: "Please authenticate using a valid token"});
    }
}

export default fetchuser;
