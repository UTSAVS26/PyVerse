import React from 'react';
import PropTypes from 'prop-types';
import { Link } from 'react-router-dom';
import '../../css/Card.css';

const Card = ({ imgSrc, title, description, path, btnName }) => {
  return (
    <div className="glass-card">
      <div className="glass-card-image-container">
        <h5 className="glass-card-title">{title}</h5>
        <hr class="solid"></hr>
        <img src={imgSrc} alt={title} className="glass-card-image" />
      </div>
      <div className="glass-card-content">
        <p className="glass-card-description">{description}</p>
        <Link to={path} className="glass-card-button">
          {btnName}
        </Link>
      </div>    
    </div>
  );
};

Card.propTypes = {
  imgSrc: PropTypes.string.isRequired,
  title: PropTypes.string.isRequired,
  description: PropTypes.string.isRequired,
  path: PropTypes.string.isRequired,
  btnName: PropTypes.string.isRequired,
};

export default Card;