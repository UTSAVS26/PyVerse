import React, { useState } from 'react';

import { getIcon } from '../../utils/icons';

import './SideBar.css';

const SideBar = ({ Actions, handleActionClick }) => {
    const [isMenuOpen, setIsMenuOpen] = useState(false);

    const toggleMenu = () => {
        setIsMenuOpen(!isMenuOpen);
    };

    return (
        <>
            <div className='menu-toggle' onClick={toggleMenu}>
                <div className='menu-icon'></div>
                <div className='menu-icon'></div>
                <div className='menu-icon'></div>
            </div>
        <div className={`sidebar-container ${isMenuOpen ? 'open' : ''}`}>
            <ul className={`sidebar-menu ${isMenuOpen ? 'open' : ''}`}>
                {Actions.map((ele, key) => {
                    const IconComponent = getIcon(ele.title);
                    return (
                        <div
                            className='actions'
                            key={key}
                            onClick={() => {
                                handleActionClick(ele.title);
                                toggleMenu();
                            }}
                        >
                            <li style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center" }}>
                                {IconComponent && <IconComponent size="30" />}
                                {ele.title}
                            </li>
                        </div>
                    );
                })}
            </ul>
        </div>
        </>
    );
};

export default SideBar;
