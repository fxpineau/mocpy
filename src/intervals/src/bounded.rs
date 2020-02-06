use num::{Integer, One, PrimInt};
use std::ops::Range;


#[inline(always)]
const fn num_bits<T>() -> usize {
    std::mem::size_of::<T>() * 8
}

const TO_EVEN_MASK: u32 = !0x1;

pub trait Bounded<T>
where
    T: Integer + PrimInt,
{
    const MAXDEPTH: i8;
    const MAXPIX: T;

    #[inline(always)]
    fn get_msb(x: T) -> u32 {
        num_bits::<T>() as u32 - x.leading_zeros() - 1
    }

    // Rename `depth_pix` to be coherent with the order of the elements in the tuple?
    // Return depth as a u8 instead of a u32?
    #[inline(always)]
    fn pix_depth(u: T) -> (u32, T) {
        let msb = Self::get_msb(u) & TO_EVEN_MASK;

        let depth = (msb >> 1) - 1;
        let t: T = One::one();
        let pix = u - t.unsigned_shl(msb);

        (depth, pix)
    }

    #[inline(always)]
    // Provide depth as a u8 instead of a u32?
    fn to_uniq(depth: u32, pix: T) -> T {
        let mut t: T = One::one();
        t = t.unsigned_shl(2);
        t.unsigned_shl(depth << 1) + pix
    }

    #[inline(always)]
    fn uniq_to_range(u: T) -> Range<T> {
        let (depth, pix) = Self::pix_depth(u);
        let tdd = (Self::MAXDEPTH as u32 - depth) << 1;
        Range {
            start: pix.unsigned_shl(tdd),
            end: (pix + One::one()).unsigned_shl(tdd) - One::one(),
        }
    }
    
}

impl Bounded<u128> for u128 {
    const MAXDEPTH: i8 = 62;
    const MAXPIX: u128 = 3 << 126;
}

impl Bounded<u64> for u64 {
    const MAXDEPTH: i8 = 29;
    const MAXPIX: u64 = 3 << 60;
}
impl Bounded<i64> for i64 {
    const MAXDEPTH: i8 = 29;
    const MAXPIX: i64 = 3 << 60;
}

impl Bounded<u32> for u32 {
    const MAXDEPTH: i8 = 13;
    const MAXPIX: u32 = 3 << 28;
}
impl Bounded<u8> for u8 {
    const MAXDEPTH: i8 = 2;
    const MAXPIX: u8 = 3 << 6;
}
