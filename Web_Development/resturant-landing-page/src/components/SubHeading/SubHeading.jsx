import React from 'react';

import { images } from '../../constants';

// SubHeading component to display a title and an accompanying image
const SubHeading = ({ title }) => (
  <div style={{ marginBottom: '1rem'}}>
    <p className="p__cormorant">{title}</p>
    <img src={images.spoon} alt="spoon" className="spoon__img" />
  </div>
);

export default SubHeading;
