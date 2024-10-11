import React, {useState} from 'react'
import Spinner from './Spinner'
import useGIF from '../hooks/useGIF';

const GIFByTag = () => {
    const [tag, setTag] = useState('');
    const { loading, fetchGIF, gif } = useGIF(tag);
    return (
        <div className='bg-purple-400 w-1/2 rounded-xl border border-black flex flex-col items-center justify-center gap-y-6 p-4'>
            <h1 className='text-2xl font-bold'>Random {tag} GIF</h1>
            {
                loading ? (<Spinner />) : (<img className='rounded-lg' src={gif} alt="Random GIF" />)
            }
            <input className='w-11/12 rounded-lg p-2 text-center text-xl' type="text" onChange={(event)=>{
                setTag(event.target.value);
            }}/>
            <button className='w-11/12 rounded-xl p-2 bg-white font-bold text-2xl' onClick={fetchGIF}>Generate</button>
        </div>
    )
}

export default GIFByTag
