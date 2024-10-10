import React from 'react'

const Filter = ({filterData,category,setCategory}) => {
  return (
    <div className='filter'>
        {filterData.map((Category)=>{
            return (<button className={`filter-btn ${category === Category.title ? 'selected' : ''}`} key={Category.id} onClick={()=>{
                setCategory(Category.title);
            }}>
                {Category.title}
            </button>)
        })}
    </div>
  )
}

export default Filter
