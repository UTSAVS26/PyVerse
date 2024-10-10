import { FaCircleUser, FaUsers, FaAddressCard, FaPowerOff } from "react-icons/fa6";

const icons = {
    "Profile": FaCircleUser,
    "Users": FaUsers,
    "Add New User": FaAddressCard,
    "Logout": FaPowerOff,
}
export function getIcon(name) {
    return icons[name];
}